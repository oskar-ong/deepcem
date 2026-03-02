import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import jsonlines
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from deepcem.serialize import serialize_to_ditto_wo_id
from deepcem.utils import get_attrs_for_keys
from evaluate import calc_metrics
from matcher import load_model, predict 

dataset = "imdb"
task = f"{dataset}_no_title"
lm = "ditto"
do_train = True

movie_dir = f"./data/processed/imdb/movie/ditto"
movie_train_fp = f"{movie_dir}/train.txt"
movie_valid_fp = f"{movie_dir}/valid.txt"
movie_test_fp = f"{movie_dir}/test.txt"

name_dir = f"./data/processed/imdb/name/ditto"
name_train_fp = f"{name_dir}/train.txt"
name_valid_fp = f"{name_dir}/valid.txt"
name_train_fp = f"{name_dir}/test.txt"

checkpoint_dir = f"./models/ditto/checkpoints/{task}"
model_path = f"./models/ditto/checkpoints/{task}/model.pt"
configs_path = f"./models/ditto/configs.json"
log_dir = f"./logs"
index_column = "id"

def main():

    with Path(configs_path).open("r", encoding="utf-8") as f:
        file_data: list = json.load(f)

    if any(entry.get("name") == task for entry in file_data):
        print("Config entry already exists")
    else:
        new_config_entry = {
            "name": task,
            "task_type": "classification",
            "vocab": ["0", "1"],
            "trainset": movie_train_fp,
            "validset": movie_valid_fp,
            "testset": movie_test_fp
        }
        file_data.append(new_config_entry)
        with Path(configs_path).open("w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4)
        print("Config Entry added")

    
    # If no model already exists -> Train
    # If flag is set -> Train
    # otherwise skip to prediction
    if not Path(model_path).exists() or do_train == True:
        print(f"Start Finetuning for task: {task}")
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
        print(f"Finished Finetuning for task: {task}")

    config, model = load_model(task, "./models/ditto/checkpoints",
                            "roberta", True, False)
    model.eval()

    predict(movie_test_fp,f"./ditto_out/{task}.jsonl", config, model,
            summarizer=None,
            max_len=128,
            lm="roberta",
            dk_injector=None,
            threshold=None)
    
    acc, prec, rec, f1 = calc_metrics(f"./ditto_out/{task}.jsonl", movie_test_fp)
    print(acc, prec, rec, f1)

if __name__=="__main__":
    main()