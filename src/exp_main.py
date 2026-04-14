import argparse
from pathlib import Path
import sqlite3
import time
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import csv
import math
import json

from ditto_wrapper import evaluate, finetune, refinetune
from experiment_config import REGISTRY, ExperimentConfig, DITTO_CONFIG
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


def run_iteration(iter_num, config: dict[str, ExperimentConfig], scores: Dict[str, Dict[Tuple[str, str], float]], relation_maps, sql_log: ExperimentLogger, log, dataset, pollution, is_bin, is_damp, seed, train_suffix, run_ids):
    start_time = time.perf_counter()
    f1_scores = {}

    # create a new folder to store scores
    # reuse log run_id
    job_id, _ = get_experiment_metadata()
    out_path = Path(
        f"./ditto_out/{dataset}/{pollution}/inference/{job_id}")
    Path(out_path).mkdir(parents=True, exist_ok=True)
    new_scores = {name: entity_dict.copy()
                  for name, entity_dict in scores.items()}

    for entity in config.values():
        # task
        if train_suffix == "":
            task = f"{entity.model_base}_{seed}_rel"
        else:
            task = f"{entity.model_base}_{seed}_{train_suffix}_rel"

        # special tokens
        special_tokens = []
        for r in entity.relations:
            special_tokens.append(r.score_col)

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
                           cp_output_fp, log, entity.true_cp_fp, entity.injected_scores_dir, special_tokens)
        metrics_cp = {"accuracy": metrics[0], "precision": metrics[1],
                      "recall": metrics[2], "f1_score": metrics[3]}
        end_cp = time.perf_counter()
        runtime_cp = end_cp - start_time

        run_id = run_ids[entity.name]

        sql_log.log_metrics(
            run_id=run_id,
            pollution=pollution,
            iteration=iter_num,
            is_final=False,
            testset="scoring",
            metrics_dict=metrics_cp,
            num_pairs=0,  # TODO Read from experiment config
            runtime=runtime_cp)

        # Track convergence
        conv_output_fp = out_path / \
            f"{entity.name}_{iter_num}_conv_results.jsonl"
        metrics = evaluate(task, conv_input_fp,
                           conv_output_fp, log, entity.true_test_fp, entity.injected_scores_dir, special_tokens)
        metrics_conv = {
            "accuracy": metrics[0], "precision": metrics[1], "recall": metrics[2], "f1_score": metrics[3]}
        f1_scores[entity.name] = metrics[3]
        end_conv = time.perf_counter()
        runtime_conv = end_conv - start_time
        sql_log.log_metrics(
            run_id=run_id,
            pollution=pollution,
            iteration=iter_num,
            is_final=False,
            testset="conv",
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
    # --- parse arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pollution", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_suffix", type=str, default="")
    parser.add_argument("--binning", action="store_true")
    parser.add_argument("--dampening", action="store_false")
    args = parser.parse_args()
    dataset = args.dataset

    # --- logging ---
    log = setup_logger("finetune")
    log.info(f"Start Finetuning: Dataset: {args.dataset}")

    sql_log = ExperimentLogger("cem_results.db")

    # --- load config ---
    raw_config = REGISTRY[dataset]
    config: dict[str, ExperimentConfig] = {
        name: entity.resolve_paths(
            args.dataset, args.pollution)
        for name, entity in raw_config.items()
    }

    # train_size
    try:
        if args.train_suffix == "":
            train_size = 1.0
        else:
            # Converts "_125" -> 125 -> 0.125 (assuming base is 1000)
            # Or adjust logic based on your specific naming convention
            train_size = round(float(args.train_suffix.strip("_")) / 1000.0, 3)
    except ValueError:
        train_size = 1.0

    run_ids = {}
    for entity in config.values():
        start_time = time.perf_counter()
        # get special tokens
        special_tokens = []
        for r in entity.relations:
            special_tokens.append(r.score_col)

        if args.train_suffix == "":
            task_base = f"{entity.model_base}_{args.seed}"
        else:
            task_base = f"{entity.model_base}_{args.seed}_{args.train_suffix}"
        # --- Phase 1: Initial Finetune, only attribute values ---
        finetune(task_base,
                 entity.empty_scores_dir, log, special_tokens, args.train_suffix, args.seed)

        input_path = f"{entity.empty_scores_dir}/test.txt"

        out_path = Path(
            f"./ditto_out/{dataset}/{args.pollution}/{entity.name}/finetune")
        Path(out_path).mkdir(parents=True, exist_ok=True)
        output_fp = f"{out_path}/phase1_eval.jsonl"
        evaluate(task_base, input_path, output_fp, log, input_path,
                 entity.empty_scores_dir, special_tokens)

        run_ids[entity.name] = sql_log.log_run(
            dataset=dataset,
            entity=entity.name,
            train_size=train_size,
            model_type="1",
            batch_size=DITTO_CONFIG['batch_size'],
            max_len=DITTO_CONFIG['max_len'],
            learning_rate=DITTO_CONFIG['learning_rate'],
            epochs=DITTO_CONFIG['epochs'],
            lm=DITTO_CONFIG['lm'],
            neg_ratio=0,  # TODO
            seed=args.seed)

        end_time = time.perf_counter()
        runtime_phase1 = end_time - start_time

        task_rel = f"{task_base}_rel"
        # --- Phase 2: Re-finetune, include relaitonal scores ---
        refinetune(task_rel,
                   entity.injected_scores_dir, task_base, log, special_tokens, args.train_suffix, args.seed)

        input_path = f"{entity.injected_scores_dir}/test.txt"

        output_fp = f"{out_path}/phase2_eval.jsonl"
        evaluate(task_rel, input_path, output_fp, log, input_path,
                 entity.injected_scores_dir, special_tokens)
        end_time = time.perf_counter()
        runtime_phase2 = end_time - start_time

    # --- start matching ---

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
    last_iter = 0
    for i in range(0, max_iters):
        log.info(f"Start iteration {i}")
        last_iter = i
        scores, f1_scores = run_iteration(
            i, config, scores, relation_maps, sql_log, log, args.dataset, args.pollution, args.binning, args.dampening, args.seed, args.train_suffix, run_ids)
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

    for entity_name, run_id in run_ids.items():
        with sqlite3.connect("cem_results.db", timeout=60) as conn:
            conn.execute("""
                UPDATE metrics 
                SET is_final = 1 
                WHERE run_id = ? AND iteration = ?
            """, (run_id, last_iter))


if __name__ == "__main__":
    main()
