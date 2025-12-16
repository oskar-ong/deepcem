import json
import os
from pathlib import Path
import shutil
import subprocess
import pandas as pd

from deepcem.clustering import bootstrap_clusters, init_pq, iterative_merge
from deepcem.config import LATE_FUSION, PipelineConfig
from deepcem.graph import create_hyperedges, extract_references
from deepcem.metrics import get_pred_labels
from deepcem.serialize import serialize_to_ditto_wo_id
from deepcem.utils import attach_run_file_handler, extract_relation, get_attrs_for_keys, setup_base_logger

dataset = "dirty/dblp-scholar"
task = f"{dataset}_lf"
lm = "ditto"
dataset_dir = f"./data/raw/deepmatcher/{dataset}"
output_dir = f"./data/processed/{lm}-splits/{dataset}"
checkpoint_dir = f"./models/ditto/checkpoints/{task}"
model_path = f"./models/ditto/checkpoints/{task}/model.pt"
configs_path = f"./models/ditto/configs.json"
log_dir = f"./logs"
index_column = "id"

if __name__=="__main__":

    train_fp = f"{dataset_dir}/train.csv" 
    train_df = pd.read_csv(train_fp)

    valid_df = pd.read_csv(f"{dataset_dir}/valid.csv")

    test_df = pd.read_csv(f"{dataset_dir}/test.csv")

    df_A = pd.read_csv(f"{dataset_dir}/tableA.csv")  # left
    df_B = pd.read_csv(f"{dataset_dir}/tableB.csv")  # right

    # Convert to string once and index by 'id'
    df_A_idx = df_A.set_index(index_column)
    df_B_idx = df_B.set_index(index_column)

    train = get_attrs_for_keys(df_A_idx, df_B_idx, train_df)
    valid = get_attrs_for_keys(df_A_idx, df_B_idx, valid_df)
    test  = get_attrs_for_keys(df_A_idx, df_B_idx, test_df)

    if not Path(f"{output_dir}/lf").exists():
        os.makedirs(f"{output_dir}/lf")

    train_ditto = serialize_to_ditto_wo_id(train)
    with open(f"{output_dir}/lf/train.txt","w", encoding="utf-8") as f:
        f.write("\n".join(train_ditto))

    valid_ditto = serialize_to_ditto_wo_id(valid)
    with open(f"{output_dir}/lf/valid.txt","w", encoding="utf-8") as f:
        f.write("\n".join(valid_ditto))
        
    test_ditto = serialize_to_ditto_wo_id(test)
    with open(f"{output_dir}/lf/test.txt","w", encoding="utf-8") as f:
        f.write("\n".join(test_ditto))

    with Path(configs_path).open("r", encoding="utf-8") as f:
        file_data: list = json.load(f)

    if any(entry.get("name") == task for entry in file_data):
        print("Entry already exists")
    else:
        new_config_entry = {
            "name": task,
            "task_type": "classification",
            "vocab": ["0", "1"],
            "trainset": f"{output_dir}/lf/train.txt",
            "validset": f"{output_dir}/lf/valid.txt",
            "testset": f"{output_dir}/lf/test.txt"
        }
        file_data.append(new_config_entry)
        with Path(configs_path).open("w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4)
        print("Entry added")

    if not Path(model_path).exists():
        print("Path does not exist")
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

    cfg = PipelineConfig()
    cfg.dataset = task
    cfg.strategy = LATE_FUSION
    cfg.alpha = 0.1
    alpha_str = str(cfg.alpha).replace('.', '-')
    threshold = 0.8
    cfg.threshold = threshold
    threshold_str = str(threshold).replace('.', '_')

    logger = setup_base_logger("cem")

    # -------------------------------------------------- # 
    # MAIN CEM ALGORITHM
    # -------------------------------------------------- # 
    candidates = pd.DataFrame(test)
    person_isAuthor_paper_test = extract_relation(test,"authors")
    relationships = person_isAuthor_paper_test

    attach_run_file_handler(logger, log_dir, alpha_str, threshold_str)

    logger.info(
        f"=== Pipeline Run | alpha={cfg.alpha}  | threshold={threshold}===")

    # --------------------------------------------------
    logger.info("Start: Extract References")
    references = extract_references(candidates)
    logger.info("Done: Extract References")
    logger.info(f"Amount References: {len(references)}")

    # --------------------------------------------------
    logger.info("Start: Create Hyperedges")
    hyperedges, references = create_hyperedges(references, relationships)
    logger.info("Done: Create Hyperedges")
    logger.info(f"Amount Hyperedges: {len(hyperedges)}")

    # --------------------------------------------------
    logger.info("Start: Bootstrap Clusters")
    clusters, parents = bootstrap_clusters(
        candidates, references, hyperedges)
    logger.info("Done: Bootstrap Clusters")
    logger.info(f"Created Clusters: {len(clusters)}")

    # --------------------------------------------------
    logger.info("Start: Initialize Priority Queue")
    pq, clusters = init_pq(clusters, references, hyperedges, parents, cfg)
    logger.info("Done: Priority Queue Initialized")
    logger.debug(pq.peek(5))

    # --------------------------------------------------
    logger.info("Start: Iterative Merge")
    clusters, references, hyperedges, parents = iterative_merge(
        pq, clusters, parents, hyperedges, references, cfg
    )
    logger.info("Done: Iterative Merge")

    # --------------------------------------------------
    # Evaluate result
    y_true = [int(l[2]) for l in test]
    acc, prec, rec, f1 = get_pred_labels(
        y_true, candidates, references, clusters, parents)

    logger.info(f"accuracy : {acc}")
    logger.info(f"precision: {prec}")
    logger.info(f"recall   : {rec}")
    logger.info(f"f1       : {f1}")

    logger.info(f"=== End Pipeline Run ===")
