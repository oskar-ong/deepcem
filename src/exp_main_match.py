import argparse
from collections import defaultdict
import csv
import time
import json
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ditto_wrapper import evaluate
from experiment_config import REGISTRY, ExperimentConfig
from logging_setup import ExperimentLogger, get_experiment_metadata, setup_logger


def build_relation_map(csv_fp: str, column1: str, column2: str) -> Dict[str, Set[str]]:
    relation_map: Dict[str, Set[str]] = defaultdict(set)
    with open(csv_fp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c1, c2 = row[column1], row[column2]
            if c1 and c2:
                relation_map[c1].add(c2)
    return dict(relation_map)


def run_iteration(iter_num, config: dict[str, ExperimentConfig], scores: Dict[str, Dict[Tuple[str, str], float]], relation_maps, sql_log: ExperimentLogger, log, dataset, pollution, is_bin, is_damp, seed, train_suffix):
    start_time = time.perf_counter()
    f1_scores = {}
    run_id, _ = get_experiment_metadata()
    out_path = Path(
        f"./ditto_out/{dataset}/{pollution}/inference/{run_id}")
    Path(out_path).mkdir(parents=True, exist_ok=True)
    new_scores = {name: entity_dict.copy()
                  for name, entity_dict in scores.items()}

    for entity in config.values():
        # task
        if train_suffix == "":
            task = f"{entity.model_base}_{seed}_rel"
        else:
            task = f"{entity.model_base}_{seed}_{train_suffix}_rel"

        # Update Scores
        cp_input_fp = out_path / f"{entity.name}_{iter_num}_input_cp.jsonl"
        # Track f1 convergenc
        conv_input_fp = out_path / f"{entity.name}_{iter_num}_input_conv.jsonl"

        # A cleaner and more performant solution would be to update both in one go, possible TODO
        update_input_files(entity.template_cp, cp_input_fp,
                           entity, relation_maps, scores, is_bin)
        update_input_files(entity.template_conv, conv_input_fp,
                           entity, relation_maps, scores, is_bin)

        # Generate new Scores
        cp_output_fp = out_path / \
            f"{entity.name}_{iter_num}_cp_results.jsonl"
        metrics = evaluate(task, cp_input_fp,
                           cp_output_fp, log, entity.true_cp_fp)
        metrics_cp = {"accuracy": metrics[0], "precision": metrics[1],
                      "recall": metrics[2], "f1_score": metrics[3]}
        end_cp = time.perf_counter()
        runtime_cp = end_cp - start_time
        sql_log.log_metrics(
            pollution=pollution,
            iteration=iter_num,
            entity=entity.name,
            testset_type="cp",
            metrics_dict=metrics_cp,
            num_pairs=0,  # TODO Read from experiment config
            runtime=runtime_cp)

        # Track convergence
        conv_output_fp = out_path / \
            f"{entity.name}_{iter_num}_conv_results.jsonl"
        metrics = evaluate(task, conv_input_fp,
                           conv_output_fp, log, entity.true_test_fp)
        metrics_conv = {
            "accuracy": metrics[0], "precision": metrics[1], "recall": metrics[2], "f1_score": metrics[3]}
        f1_scores[entity.name] = metrics[3]
        end_conv = time.perf_counter()
        runtime_conv = end_conv - start_time
        sql_log.log_metrics(
            pollution=pollution,
            iteration=iter_num,
            entity=entity.name,
            testset_type="conv",
            metrics_dict=metrics_conv,
            num_pairs=0,  # TODO
            runtime=runtime_conv)

        # Update Scores Map
        new_scores[entity.name] = extract_scores(
            cp_output_fp, new_scores[entity.name], is_damp)

    return new_scores, f1_scores


def extract_scores(fp, dependency_scores, is_damp=False):
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            left_id = data['left']['id']
            right_id = data['right']['id']
            key = tuple(sorted((left_id, right_id)))
            match = int(data['match'])
            confidence = data['match_confidence']

            if match == 1:
                if is_damp == True:
                    confidence = (
                        0.2 * dependency_scores[key]) + (0.8 * confidence)
                else:
                    dependency_scores[key] = confidence
            elif match == 0:
                if is_damp == True:
                    confidence = (
                        0.2 * dependency_scores[key]) + (0.8 * (1-confidence))
                else:
                    dependency_scores[key] = (1 - confidence)
    return dependency_scores


def extract_pairs(fp):
    pairs = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            left_id = data[0]['id']
            right_id = data[1]['id']

            pairs.append((left_id, right_id))
    return pairs


def update_input_files(template_fp, out_fp, entity_cfg: ExperimentConfig, relationship_maps, all_scores, is_bin=False):
    threshold = 0.15
    with open(template_fp, 'r', encoding='utf-8') as infile, open(out_fp, 'w', encoding='utf-8') as outfile:
        for line in infile:
            record_pair = json.loads(line.strip())

            if len(record_pair) >= 2:
                left_id = record_pair[0].get('id')
                right_id = record_pair[1].get('id')

                for r in entity_cfg.relations:
                    relation_map = relationship_maps[f"{entity_cfg.name}{r.name}"]
                    related_scores = all_scores[r.name]

                    score = calc_monge_elkan(
                        left_id, right_id, relation_map, related_scores)

                    # BINNING
                    if is_bin == True:
                        if score >= 0.85:
                            score = "HIGH"
                        if score <= 0.15:
                            score = "LOW"
                        if 0.15 < score < 0.85:
                            score = "UNC"  # uncertain
                    else:
                        # is score meaningful enough? If too fuzzy, ignore
                        if abs(score - 0.5) < threshold:
                            score = 0.5

                    record_pair[0][r.score_col] = score
                    record_pair[1][r.score_col] = score

            outfile.write(json.dumps(record_pair) + '\n')


def calc_monge_elkan(left_id, right_id, relationship_map: Dict[str, List[str]], dependency_scores):

    dependencies_left = relationship_map.get(left_id, set())
    dependencies_right = relationship_map.get(right_id, set())

    if len(dependencies_right) < len(dependencies_left):
        tmp = dependencies_right
        dependencies_right = dependencies_left
        dependencies_left = tmp

    scores = []

    if dependencies_left and dependencies_right:
        for dep_left in dependencies_left:
            c_max = 0.0  # current max score for this dependency

            # If one of the left dependencies is same key as right, then max score = 1
            if dep_left in dependencies_right:
                scores.append(1.0)
                continue
            for dep_right in dependencies_right:
                score = dependency_scores.get(
                    tuple(sorted((dep_left, dep_right))), 0.5)

                if score > c_max:
                    c_max = score
            scores.append(c_max)

        monge_elkan = (1/len(dependencies_left)) * sum(scores)
        return round(monge_elkan, 2)
    else:
        return 0.5


def main():
    # --- arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pollution", type=str)
    parser.add_argument("--binning", action="store_true")
    parser.add_argument("--dampening", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_suffix", type=str, default="")
    args = parser.parse_args()

    # --- logs ---
    log = setup_logger("matching")
    log.info(f"Start Collective Entity Matching: Dataset: {args.dataset}")
    sql_log = ExperimentLogger("cem_results.db")

    # stop after max iterations
    max_iters = 4

    # --- read and resolve configs ---
    raw_config = REGISTRY[args.dataset]
    config = {
        name: entity.resolve_paths(
            args.dataset, args.pollution)
        for name, entity in raw_config.items()
    }

    # --- initialize scores ---
    scores_init: Dict[str, Dict[Tuple[str, str], float]] = {}
    relation_maps = {}
    log.info(f"Initialize Scores for each pair...")
    for entity in config.values():
        # Pairs are for Score Generation -> CP
        pairs = extract_pairs(entity.template_cp)
        scores_init[entity.name] = {tuple(sorted(pair)): 0.5 for pair in pairs}
        for r in entity.relations:
            relation_maps[f"{entity.name}{r.name}"] = build_relation_map(
                r.junction_table, entity.id_col, r.fk)
    log.info(f"Score Initialization Done!")

    # --- Start Matching ---
    old_f1_scores = defaultdict(int)
    log.info(f"Start iterative matching")
    scores = scores_init
    for i in range(0, max_iters):
        log.info(f"Start iteration {i}")
        scores, f1_scores = run_iteration(
            i, config, scores, relation_maps, sql_log, log, args.dataset, args.pollution, args.binning, args.dampening, args.seed, args.train_suffix)
        converged = True
        for k in f1_scores.keys():

            # compare floats with math module
            if not math.isclose(f1_scores[k], old_f1_scores[k], rel_tol=1e-5):
                converged = False

        if converged == True:
            break
        old_f1_scores = f1_scores
        log.info(f"Finished iteration {i}")
        log.info(f"-------------------------")


if __name__ == "__main__":
    main()
