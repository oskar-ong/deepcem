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
from matcher import load_model, predict 

dataset = "imdb"
task = f"{dataset}_base_no_author"
lm = "ditto"


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

def to_str(ent1, ent2, summarizer=None, max_len=256):
    """Serialize a pair of data entries

    Args:
        ent1 (Dictionary): the 1st data entry
        ent2 (Dictionary): the 2nd data entry
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length

    Returns:
        string: the serialized version
    """
    content = ''
    for ent in [ent1, ent2]:
        if isinstance(ent, str):
            content += ent
        else:
            for attr in ent.keys():
                content += 'COL %s VAL %s ' % (attr, ent[attr])
        content += '\t'

    content += '0'

    if summarizer is not None:
        content = summarizer.transform(content, max_len=max_len)

    new_ent1, new_ent2, _ = content.split('\t')

    return new_ent1 + '\t' + new_ent2 + '\t0'

def main():

    with Path(configs_path).open("r", encoding="utf-8") as f:
        file_data: list = json.load(f)

    if any(entry.get("name") == task for entry in file_data):
        print("Entry already exists")
    else:
        new_config_entry = {
            "name": task,
            "task_type": "classification",
            "vocab": ["0", "1"],
            "trainset": f"{movie_train_fp}",
            "validset": f"{movie_valid_fp}",
            "testset": f"{movie_test_fp}"
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

    config, model = load_model(task, "./models/ditto/checkpoints",
                            "roberta", True, False)
    model.eval()

    predict(movie_test_fp,"./ditto_out", config, model,
            summarizer=None,
            max_len=128,
            lm="roberta",
            dk_injector=None,
            threshold=None)

if __name__=="__main__":
    main()