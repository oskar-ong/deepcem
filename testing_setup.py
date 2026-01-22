
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd
from deepcem.clustering import UnionFind, bootstrap_clusters, init_pq, iterative_merge
from deepcem.config import LATE_FUSION, AlgoConfig
from deepcem.graph import create_hyperedges, extract_references
from deepcem.metrics import get_pred_labels
from deepcem.preprocessing import build_block_index, build_ditto_file_from_labels, enrich_pairs_dataframe, match_context_entities, normalize, normalize_simple_context, reduce, save_as_ditto_file
from deepcem.utils import attach_run_file_handler, setup_base_logger

dataset = "dirty/dblp-scholar"
task = "testing-setup"
lm = "ditto"
dataset_dir = f"./data/raw/deepmatcher/{dataset}"
output_dir = f"./data/processed/{lm}-splits/testing_setup"
checkpoint_dir = f"./models/ditto/checkpoints/{task}"
model_path = f"./models/ditto/checkpoints/{task}/model.pt"
configs_path = f"./models/ditto/configs.json"
log_dir = f"./logs"
index_column = "id" 
finetune = True


cfg = AlgoConfig()
cfg.dataset = task
cfg.strategy = LATE_FUSION
cfg.use_gpu = True
cfg.fp16 = True

logger = setup_base_logger("cem")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    
    args = parser.parse_args()

    threshold = args.threshold
    cfg.threshold = threshold
    threshold_str = str(threshold).replace('.', '_')

    cfg.alpha = args.alpha
    alpha_str = str(cfg.alpha).replace('.', '-')
    
    print(f"Starting experiment with Threshold: {cfg.threshold} and Alpha: {cfg.alpha}")

    splits = ["train", "valid", "test"]
    table_configs = [("a", "ltable_id"), ("b", "rtable_id")]
    pairs = {}
    enriched_pairs = {}
    candidates = {}

    # Access via: results['train']['a']['authors']
    results = {split: {} for split in splits}

    for split in splits:
        split_fp = f"{dataset_dir}/{split}.csv"
        
        for suffix, id_col in table_configs:
            reduced_fp = f"./data/processed/reduced/{split}_{suffix}.csv"
            table_raw_fp = f"{dataset_dir}/table{suffix.upper()}.csv"
            
            # 1. Reduce
            reduce(split_fp, table_raw_fp, reduced_fp, "id", id_col)
            
            # 2. Normalize and save all 4 attributes in the results dict
            norm_data = normalize(reduced_fp, "id", "authors", ",")
            
            results[split][suffix] = {
                "publications": norm_data[0],
                "authors":      norm_data[1],
                "pub_to_auth":  norm_data[2],
                "auth_to_pub":  norm_data[3]
            }
            
            print(f"{split} {suffix} count: {len(norm_data[0])}")

        pairs[split] = pd.read_csv(split_fp, dtype={'left': str, 'right': str})
        
        # 3. Build Ditto file
        enriched_pairs[split] = enrich_pairs_dataframe(pairs[split], results[split]["a"]["publications"], results[split]["b"]["publications"])

        save_as_ditto_file(enriched_pairs[split], output_dir,split)
    
    # fine tune 
    if finetune:
        with Path(configs_path).open("r", encoding="utf-8") as f:
            file_data: list = json.load(f)

        if any(entry.get("name") == task for entry in file_data):
            print("Entry already exists")
        else:
            new_config_entry = {
                "name": task,
                "task_type": "classification",
                "vocab": ["0", "1"],
                "trainset": f"{output_dir}/train.txt",
                "validset": f"{output_dir}/valid.txt",
                "testset": f"{output_dir}/test.txt"
            }
            file_data.append(new_config_entry)
            with Path(configs_path).open("w", encoding="utf-8") as f:
                json.dump(file_data, f, indent=4)
            print("Entry added")

        if not Path(model_path).exists():
            print("Path does not exist. Fine Tune")
            shutil.copyfile(configs_path, 'configs.json')
            cmd = [
                "python",
                f"./models/{lm}/train_ditto.py",
                "--task", task,
                "--batch_size", "32",
                "--max_len", "128",
                "--lr", "3e-5",
                "--n_epochs", "1",
                "--finetuning",
                "--lm", "roberta",
                "--fp16",
                "--save_model",
                "--logdir", "./models/ditto/checkpoints/",
            ]

            env = os.environ.copy()
            #env["CUDA_VISIBLE_DEVICES"] = "0"

            subprocess.run(cmd, env=env)

    # cluster all authors (authors_a and authors_b) via exact_match other conservative matching 
    all_authors: dict = results['test']['a']['authors'] | results['test']['b']['authors']
    print(len(results['test']['a']['authors']))
    print(len(results['test']['b']['authors']))
    print(len(all_authors))

    for k in all_authors.keys():
        all_authors[k]['normalized'] = normalize_simple_context(all_authors[k]['name'])

    #print(all_authors.values())

    blocks = build_block_index(all_authors)

    clusters_authors = UnionFind()
    for author in all_authors:
        clusters_authors.add(author)

    print(f"Before matching authors: {len(clusters_authors.get_sets())}")
    clusters_authors = match_context_entities(all_authors, blocks,clusters_authors)
    print(f"After matching authors: {len(clusters_authors.get_sets())}")

    mention_to_cluster = {}
    for mention in all_authors:
        mention_to_cluster[mention] = clusters_authors.find(mention)

    print(len(mention_to_cluster))

    pub_to_authors = results['test']['a']['pub_to_auth'] | results['test']['b']['pub_to_auth']

    tmp_pub_to_authors = {}
    for pub, mentions in pub_to_authors.items():
        clusters = set()
        for m in mentions:
            if m in mention_to_cluster:
                clusters.add(mention_to_cluster[m])
        tmp_pub_to_authors[pub] = list(clusters)

    
    pub_to_authors = tmp_pub_to_authors
    author_to_pubs: dict[str, list] = {}

    for pub, clusters in pub_to_authors.items():
        for c in clusters:
            author_to_pubs.setdefault(c, []).append(pub)

     # -------------------------------------------------- # 
    # MAIN CEM ALGORITHM
    # -------------------------------------------------- # 
    attach_run_file_handler(logger, log_dir, alpha_str, threshold_str)
    
    logger.info(
        f"=== Pipeline Run | alpha={cfg.alpha}  | threshold={threshold}===")

    cs = pd.DataFrame(enriched_pairs['test'])
    references = extract_references(cs)

    hyperedges = pub_to_authors

    clusters, parents = bootstrap_clusters(
        cs, references, hyperedges, author_to_pubs)

    pq, clusters = init_pq(clusters, references, hyperedges, parents, cfg)

    clusters, references, hyperedges, parents = iterative_merge(
        pq, clusters, parents, hyperedges, references, cfg
    )

    # Evaluate result
    y_true = pairs['test']['label'].astype(int).tolist()
    acc, prec, rec, f1 = get_pred_labels(
        y_true, cs, references, clusters, parents, index_column)

    logger.info(f"accuracy : {acc}")
    logger.info(f"precision: {prec}")
    logger.info(f"recall   : {rec}")
    logger.info(f"f1       : {f1}")

    logger.info(f"=== End Pipeline Run ===")

if __name__=="__main__":
    main()