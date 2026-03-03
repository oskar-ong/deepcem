from collections import defaultdict
import csv
import os
import subprocess
import json
from typing import Dict, List, Set

import pandas as pd

from evaluate import calc_metrics

def build_relation_map(csv_fp: str, column1: str, column2: str) -> Dict[str, Set[str]]:
    relation_map: Dict[str, Set[str]] = defaultdict(set)
    with open(csv_fp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c1, c2 = row[column1], row[column2]
            if c1 and c2:
                relation_map[c1].add(c2)
    return dict(relation_map)

# Dependency map: Dict[movie_ids: str, Dict[entity_type, List[ids: str]]]


def run_iteration(iter_num, MODELS, movie_input_template, name_input_template, movie_pairs_score, name_pairs_score, movie_table_key, name_table_key, movie_dependency_map, name_dependency_map):
    # MOVIES (MAIN ENTITY FIRST)

    # Update Paper JSONL with Venue scores from previous iter
    update_input_files(movie_input_template, movie_dependency_map, name_pairs_score, f"movie_{iter_num}_input.jsonl", movie_table_key)
    
    # Run Ditto Inference
    # Command from Ditto Docs
    #     python matcher.py \
    # --task wdc_all_small \
    # --input_path input/input_small.jsonl \
    # --output_path output/output_small.jsonl \
    # --lm distilbert \
    # --max_len 64 \
    # --use_gpu \
    # --fp16 \
    # --checkpoint_path checkpoints/
    cmd = [
            "python",
            f"./models/ditto/matcher.py",
            "--task", MODELS["movies"],
            "--input_path", f"movie_{iter_num}_input.jsonl",
            "--output_path", f"ditto_out/movie_{iter_num}.jsonl",
            "--lm", "roberta",
            "--max_len", "128",
            "--use_gpu",
            "--fp16",
            "--checkpoint_path", "./models/ditto/checkpoints/",
        ]

    env = os.environ.copy()
    #env["CUDA_VISIBLE_DEVICES"] = "0"

    subprocess.run(cmd, env=env)
    
    # Update STATE with new Paper results
    #STATE["paper_pairs"] = extract_scores(f"results_paper_{iter_num}.jsonl")
    movie_pairs_score = extract_scores(f"ditto_out/movie_{iter_num}.jsonl", movie_pairs_score, movie_table_key)
    movie_testset_fp = f"./data/processed/imdb/movie/ditto/test.txt"
    acc, prec, rec, f1 = calc_metrics(f"ditto_out/movie_{iter_num}.jsonl", movie_testset_fp)
    print(f"MOVIE METRICS FOR ITERATION {iter_num}", acc, prec, rec, f1)

    # NAMES (DEPENDENCY ENTITY)

    update_input_files(name_input_template, name_dependency_map, movie_pairs_score, f"name_{iter_num}_input.jsonl", name_table_key)
    
    cmd = [
            "python",
            f"./models/ditto/matcher.py",
            "--task", MODELS["names"],
            "--input_path", f"name_{iter_num}_input.jsonl",
            "--output_path", f"ditto_out/name_{iter_num}.jsonl",
            "--lm", "roberta",
            "--max_len", "128",
            "--use_gpu",
            "--fp16",
            "--checkpoint_path", "./models/ditto/checkpoints/",
        ]

    env = os.environ.copy()
    #env["CUDA_VISIBLE_DEVICES"] = "0"

    subprocess.run(cmd, env=env)
    
    name_pairs_score = extract_scores(f"ditto_out/name_{iter_num}.jsonl", name_pairs_score, name_table_key)

    name_testset_fp = f"./data/processed/imdb/name/ditto/test.txt"
    acc, prec, rec, f1 = calc_metrics(f"ditto_out/name_{iter_num}.jsonl", name_testset_fp)
    print(f"NAME METRICS FOR ITERATION {iter_num}", acc, prec, rec, f1)

def extract_scores(fp, dependency_scores, id_attribute):
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            left_id = data['left'][id_attribute]
            right_id = data['right'][id_attribute]
            match = int(data['match'])
            confidence = data['match_confidence']
            
            if match == 1:
                dependency_scores[(left_id, right_id)] = confidence
            elif match == 0:
                dependency_scores[(left_id, right_id)] = (1 - confidence) 
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

def update_input_files(input_template, relationship_map, dependency_scores, output_json_fp, table_key):
    with open(input_template, 'r') as infile, open(output_json_fp, 'w') as outfile:
        for line in infile:
            # 1. Parse the line (a list of two dictionaries)
            record_pair = json.loads(line.strip())
            
            if len(record_pair) >= 2:
                # 2. Extract tconst values
                left_id = record_pair[0].get(table_key)
                right_id = record_pair[1].get(table_key)
                
                # 3. Calculate the score
                score = aggregate_dependency_scores(left_id, right_id, relationship_map, dependency_scores)
                
                # 4. Inject the score back into both objects
                record_pair[0]["REL_SCORE"] = score
                record_pair[1]["REL_SCORE"] = score
                
            # 5. Write the modified list back as a single line
            json.dump(record_pair, outfile)
            outfile.write('\n')

def aggregate_dependency_scores(left_id, right_id, relationship_map: Dict[str, List[str]], dependency_scores):

    dependencies_left = relationship_map.get(left_id, set())
    dependencies_right = relationship_map.get(right_id, set())

    all_possible_scores = []
    
    if dependencies_left and dependencies_right:
        for dep_left in dependencies_left:
            for dep_right in dependencies_right:
                score = dependency_scores.get(tuple(sorted((dep_left, dep_right))), 0.5)
                all_possible_scores.append(score)
        
    return max(all_possible_scores) if all_possible_scores else 0.5

def main():
    # Configuration
    MODELS = {"movies": "imdb_movies_rel_score", "names": "imdb_names_rel_score"}

    movie_input_template = "./data/processed/imdb/movie/input_template.jsonl"
    name_input_template = "./data/processed/imdb/name/input_template.jsonl"

    # movie_pairs = list(zip(movie_test_df[('left', 'tconst')], movie_test_df[('right', 'tconst')])) # list of pairs from movie test set
    movie_table_key = "tconst"
    movie_pairs = extract_pairs(movie_input_template, movie_table_key)
    movie_pairs_score = dict.fromkeys(movie_pairs, 0.5)
    # name_test_df = pd.read_parquet('data_prep/name_test_rel_score.parquet')
    name_table_key = "nconst"
    name_pairs = extract_pairs(name_input_template, name_table_key) # list of pairs from name test set
    name_pairs_score = dict.fromkeys(name_pairs, 0.5)

    PATH_RAW_PRINCIPALS = "./data/raw/imdb/title_principals.csv"
    movie_dependency_map = build_relation_map(PATH_RAW_PRINCIPALS, movie_table_key, name_table_key)
    name_dependency_map = build_relation_map(PATH_RAW_PRINCIPALS, name_table_key, movie_table_key)
    run_iteration(0, MODELS, movie_input_template, name_input_template, movie_pairs_score, name_pairs_score, movie_table_key, name_table_key, movie_dependency_map, name_dependency_map)
    run_iteration(1, MODELS, movie_input_template, name_input_template, movie_pairs_score, name_pairs_score, movie_table_key, name_table_key, movie_dependency_map, name_dependency_map)

if __name__=="__main__":
    main()