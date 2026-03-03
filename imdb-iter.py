from collections import defaultdict
import csv
import subprocess
import json
from typing import Dict, List, Set

import pandas as pd

# Configuration
MODELS = {"movies": "output/paper_relational/", "venue": "output/venue_relational/"}

movie_input_template = "./data/processed/imdb/movie/test_no_label.jsonl"
name_input_template = "./data/processed/imdb/name/test_no_label.jsonl"

movie_test_df = pd.read_parquet('data_prep/movie_test_rel_score.parquet')
movie_pairs = list(zip(movie_test_df[('left', 'tconst')], movie_test_df[('right', 'tconst')])) # list of pairs from movie test set
movie_pairs_score = dict.fromkeys(movie_pairs, 0.5)
movie_table_key = "tconst"
name_test_df = pd.read_parquet('data_prep/name_test_rel_score.parquet')
name_pairs = list(zip(name_test_df[('left', 'nconst')], name_test_df[('right', 'nconst')])) # list of pairs from name test set
name_pairs_score = dict.fromkeys(name_pairs, 0.5)
name_table_key = "nconst"

PATH_RAW_PRINCIPALS = "./data/raw/imdb/title_principals.csv"
#STATE = {movie_pairs_score, name_pairs_score} # {pair_id: 0.5}

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
movie_dependency_map = build_relation_map(PATH_RAW_PRINCIPALS, movie_table_key, name_table_key)


def run_iteration(iter_num):
    # MOVIES (MAIN ENTITY FIRST)

    # 1. Update Paper JSONL with Venue scores from previous iter
    update_input_files(movie_input_template, movie_dependency_map, name_pairs_score, f"movie_{iter_num}_input.jsonl", movie_table_key)
    
    # 2. Run Ditto Inference
    #subprocess.run(["python", "matcher.py", "--task", "paper", "--checkpoint", MODELS["paper"], "--input", "papers_to_match.jsonl", "--output", f"results_paper_{iter_num}.jsonl"])
    
    # 3. Update STATE with new Paper results
    #STATE["paper_pairs"] = extract_scores(f"results_paper_{iter_num}.jsonl")
    
    # 4. Repeat for Venue (using the updated Paper scores)
    # ...

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


run_iteration(1)
