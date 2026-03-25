import argparse
from collections import defaultdict
import csv
import json
from typing import Dict, List, Set

from ditto_wrapper import evaluate
from experiment_config import REGISTRY, EntityConfig
from logging_setup import ExperimentLogger, setup_logger

log = setup_logger("exp_baseline_iterative_imdb_hard-matching")
log.info("Start Experiment: Iterative Matching - IMDB HARD - matching")

def build_relation_map(csv_fp: str, column1: str, column2: str) -> Dict[str, Set[str]]:
    relation_map: Dict[str, Set[str]] = defaultdict(set)
    with open(csv_fp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c1, c2 = row[column1], row[column2]
            if c1 and c2:
                relation_map[c1].add(c2)
    return dict(relation_map)

def run_iteration(iter_num, config: dict[str, EntityConfig], scores, relation_maps, sql_log: ExperimentLogger, run_id):

    for entity in config.values():
        cp_input_fp = f"{entity.name}_{iter_num}_input_cp.jsonl" # Update Scores
        conv_input_fp = f"{entity.name}_{iter_num}_input_conv.jsonl" # Track f1 convergenc

        # A cleaner and more performant solution would be to update both in one go, possible TODO
        update_input_files(entity.template_cp, cp_input_fp, entity, relation_maps, scores, is_bin=False)
        update_input_files(entity.template_conv, conv_input_fp, entity, relation_maps, scores, is_bin=False)
                           
        # Generate new Scores
        cp_output_fp = f"ditto_out/{entity.name}_{iter_num}_cp.jsonl"
        metrics = evaluate(entity.model, cp_input_fp, cp_output_fp, log, entity.true_cp_fp)
        metrics_cp = {"accuracy": metrics[0], "precision": metrics[1], "recall": metrics[2], "f1_score": metrics[3]}
        sql_log.log_metrics(run_id, iter_num, entity.name, metrics_cp)

        # Track convergence
        conv_output_fp = f"ditto_out/{entity.name}_{iter_num}_conv.jsonl"
        metrics = evaluate(entity.model, conv_input_fp, conv_output_fp, log, entity.true_test_fp)
        metrics_conv = {"accuracy": metrics[0], "precision": metrics[1], "recall": metrics[2], "f1_score": metrics[3]}
        sql_log.log_metrics(run_id, iter_num, entity.name, metrics_conv)

    # Update Scores Map
    # Start new Loop -> Synchronous Update
    for entity in config.values():
        scores[entity.name] = extract_scores(f"ditto_out/{entity.name}_{iter_num}_cp.jsonl", scores[entity.name], entity.id_col)

    return scores, metrics_conv

def extract_scores(fp, dependency_scores, id_attribute, is_damp=False):
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            left_id = data['left'][id_attribute]
            right_id = data['right'][id_attribute]
            key = tuple(sorted((left_id, right_id)))
            match = int(data['match'])
            confidence = data['match_confidence']
            
            if match == 1:
                if is_damp == True:
                    confidence = (0.2 * dependency_scores[key]) + (0.8 * confidence)
                else:
                    dependency_scores[key] = confidence
            elif match == 0:
                if is_damp == True:
                    confidence = (0.2 * dependency_scores[key]) + (0.8 * (1-confidence))
                else:
                    dependency_scores[key] = (1 - confidence) 
    return dependency_scores

def extract_pairs(fp, id_attribute):
    pairs = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            left_id = data[0][id_attribute]
            right_id = data[1][id_attribute]

            pairs.append((left_id, right_id))
    return pairs

def update_input_files(template_fp, out_fp, entity_cfg: EntityConfig, relationship_maps, all_scores, is_bin=False):
    threshold = 0.15
    with open(template_fp, 'r') as infile, open(out_fp, 'w') as outfile:
        for line in infile:
            record_pair = json.loads(line.strip())
            
            if len(record_pair) >= 2:
                left_id = record_pair[0].get(entity_cfg.id_col)
                right_id = record_pair[1].get(entity_cfg.id_col)

                for r in entity_cfg.relations:
                    relation_map = relationship_maps[f"{entity_cfg.name}{r.name}"]
                    related_scores = all_scores[r.name]
                
                    score = calc_monge_elkan(left_id, right_id, relation_map, related_scores)

                    # BINNING
                    if is_bin == True:
                        if score >= 0.85: 
                            score = "HIGH"
                        if score <= 0.15:
                            score = "LOW"
                        if 0.15 < score < 0.85:
                            score = "UNC" # uncertain
                    else:
                        # is score meaningful enough? If too fuzzy, ignore
                        if abs(score - 0.5) < threshold:
                            score = 0.5 
                    
                    record_pair[0][r.score_col] = score
                    record_pair[1][r.score_col] = score
                
            json.dump(record_pair, outfile)
            outfile.write('\n')

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
            c_max = 0.0 # current max score for this dependency
            for dep_right in dependencies_right:
                score = dependency_scores.get(tuple(sorted((dep_left, dep_right))), 0.5)

                if score > c_max:
                    c_max = score
            scores.append(c_max)
        
        monge_elkan = ( 1/len(dependencies_left) ) * sum(scores) 
        return round(monge_elkan,2)
    else:
        return 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    sql_log = ExperimentLogger("entity_resolution_results.db")
    run_params = {'model': 'Ditto', 'lr': 3e-5, 'batch_size': 16}
    run_id = sql_log.log_run(run_params)
    prev_f1 = 0.0
    max_iters = 4

    config = REGISTRY[args.dataset]
    scores = {}
    relation_maps = {}
    for entity in config.values():
        # Pairs are for Score Generation -> CP 
        pairs = extract_pairs(entity.template_cp, entity.id_col)
        scores[entity.name] = {tuple(sorted(pair)): 0.5 for pair in pairs}
        for r in entity.relations:
            relation_maps[f"{entity.name}{r.name}"] = build_relation_map(r.junction_table, entity.id_col, r.fk)

    for i in range(0, max_iters):
        scores, metrics = run_iteration(i, config, scores, relation_maps, sql_log, run_id)

        if metrics["f1_score"] == prev_f1:
            break
        
        prev_f1 = metrics["f1_score"]

if __name__=="__main__":
    main()