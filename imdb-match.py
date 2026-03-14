from collections import defaultdict
import csv
import json
from typing import Dict, List, Set

from ditto_wrapper import evaluate
from logging_setup import setup_logger

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

def run_iteration(iter_num, MODELS, movie_input_template, name_input_template, movie_pairs_score, name_pairs_score, movie_table_key, name_table_key, movie_dependency_map, name_dependency_map):
    # ================================================================================
    # MOVIES
    # ================================================================================
    entity = "movie"
    input_path = f"{entity}_{iter_num}_input.jsonl"
    update_input_files(movie_input_template, movie_dependency_map, name_pairs_score, input_path, movie_table_key)
    
    output_path_name = f"ditto_out/{entity}_{iter_num}.jsonl"
    true_movie_inference = f"./data/imdb_hard/{entity}/inference.jsonl"
    movie_testset_fp = f"./data/imdb_hard/movie/test.txt"
    evaluate(MODELS[entity], input_path, output_path_name, "", log, true_movie_inference)

    # ================================================================================
    # NAMES
    # ================================================================================
    input_path = f"name_{iter_num}_input.jsonl"
    update_input_files(name_input_template, name_dependency_map, movie_pairs_score, input_path, name_table_key)
    
    output_path_name = f"ditto_out/name_{iter_num}.jsonl"
    true_name_inference = "./data/imdb_hard/name/inference.jsonl"
    evaluate(MODELS["names"], input_path, output_path_name, "", log, true_name_inference)

    # ================================================================================
    # UPDATE SCORES 
    # ================================================================================
    movie_pairs_score = extract_scores(f"ditto_out/movie_{iter_num}.jsonl", movie_pairs_score, movie_table_key)
    name_pairs_score = extract_scores(output_path_name, name_pairs_score, name_table_key)

    return movie_pairs_score, name_pairs_score

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

def update_input_files(input_template, relationship_map, dependency_scores, output_json_fp, table_key, is_bin=False):
    threshold = 0.15
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
                
                # 4. Inject the score back into both objects
                record_pair[0]["REL_SCORE"] = score
                record_pair[1]["REL_SCORE"] = score
                
            # 5. Write the modified list back as a single line
            json.dump(record_pair, outfile)
            outfile.write('\n')

def aggregate_dependency_scores(left_id, right_id, relationship_map: Dict[str, List[str]], dependency_scores):

    dependencies_left = relationship_map.get(left_id, set())
    dependencies_right = relationship_map.get(right_id, set())

    # Switch places so the neighborhood with fewer entries is always the left one
    # Is this necessary? 
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
    # Configuration
    MODELS = {"movies": "imdb_movies_rel_score", "names": "imdb_names_rel_score"}
    dir = "./data/imdb_hard/"
    movie_input_template = f"{dir}movie/input_template.jsonl"
    name_input_template = f"{dir}/name/input_template.jsonl"

    # movie_pairs = list(zip(movie_test_df[('left', 'tconst')], movie_test_df[('right', 'tconst')])) # list of pairs from movie test set
    movie_table_key = "tconst"
    movie_pairs = extract_pairs(movie_input_template, movie_table_key)
    # movie_pairs_score = dict.fromkeys(movie_pairs, 0.5)
    movie_pairs_score = {tuple(sorted(pair)): 0.5 for pair in movie_pairs}
    # name_test_df = pd.read_parquet('data_prep/name_test_rel_score.parquet')
    name_table_key = "nconst"
    name_pairs = extract_pairs(name_input_template, name_table_key) # list of pairs from name test set
    # name_pairs_score = dict.fromkeys(name_pairs, 0.5)
    name_pairs_score = {tuple(sorted(pair)): 0.5 for pair in name_pairs}

    PATH_RAW_PRINCIPALS = f"{dir}/title_principals.csv"
    # Dependency map: Dict[movie_ids: str, Dict[entity_type, List[ids: str]]]
    movie_dependency_map = build_relation_map(PATH_RAW_PRINCIPALS, movie_table_key, name_table_key)
    name_dependency_map = build_relation_map(PATH_RAW_PRINCIPALS, name_table_key, movie_table_key)
    movie_pairs_score, name_pairs_score = run_iteration(0, MODELS, movie_input_template, name_input_template, movie_pairs_score, name_pairs_score, movie_table_key, name_table_key, movie_dependency_map, name_dependency_map)
    movie_pairs_score, name_pairs_score = run_iteration(1, MODELS, movie_input_template, name_input_template, movie_pairs_score, name_pairs_score, movie_table_key, name_table_key, movie_dependency_map, name_dependency_map)
    movie_pairs_score, name_pairs_score = run_iteration(2, MODELS, movie_input_template, name_input_template, movie_pairs_score, name_pairs_score, movie_table_key, name_table_key, movie_dependency_map, name_dependency_map)
    movie_pairs_score, name_pairs_score = run_iteration(3, MODELS, movie_input_template, name_input_template, movie_pairs_score, name_pairs_score, movie_table_key, name_table_key, movie_dependency_map, name_dependency_map)

if __name__=="__main__":
    main()