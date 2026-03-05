import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import jsonlines
import pandas as pd
from evaluate import calc_metrics
from matcher import load_model, predict 

def main():

    # FINE TUNE MOVIE TABLE
    dataset = "imdb_hard"
    task_movie = f"{dataset}_movies"
    lm = "ditto"
    do_train = True

    movie_dir = f"./data/{dataset}/movie"
    movie_train_fp = f"{movie_dir}/train.txt"
    movie_valid_fp = f"{movie_dir}/valid.txt"
    movie_test_fp = f"{movie_dir}/test.txt"

    checkpoint_dir = f"./models/ditto/checkpoints/{task_movie}"
    model_path = f"./models/ditto/checkpoints/{task_movie}/model.pt"
    configs_path = f"./models/ditto/configs.json"
    log_dir = f"./logs"
    index_column = "id"

    with Path(configs_path).open("r", encoding="utf-8") as f:
        file_data: list = json.load(f)

    if any(entry.get("name") == task_movie for entry in file_data):
        print("Config entry already exists")
    else:
        new_config_entry = {
            "name": task_movie,
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
        print(f"Start Finetuning for task: {task_movie}")
        shutil.copyfile(configs_path, 'configs.json')
        cmd = [
            "python",
            f"./models/{lm}/train_ditto.py",
            "--task", task_movie,
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
        print(f"Finished Finetuning for task: {task_movie}")

    config, model = load_model(task_movie, "./models/ditto/checkpoints",
                            "roberta", True, False)
    model.eval()

    predict(movie_test_fp,f"./ditto_out/{task_movie}.jsonl", config, model,
            summarizer=None,
            max_len=128,
            lm="roberta",
            dk_injector=None,
            threshold=None)
    
    acc, prec, rec, f1 = calc_metrics(f"./ditto_out/{task_movie}.jsonl", movie_test_fp)
    print(task_movie, acc, prec, rec, f1)


    # FINE TUNE NAME TABLE
    task_name = f"{dataset}_names"
    lm = "ditto"
    do_train = True
    
    name_dir = f"./data/{dataset}/name"
    name_train_fp = f"{name_dir}/train.txt"
    name_valid_fp = f"{name_dir}/valid.txt"
    name_test_fp = f"{name_dir}/test.txt"

    model_path = f"./models/ditto/checkpoints/{task_name}/model.pt"
    configs_path = f"./models/ditto/configs.json"

    with Path(configs_path).open("r", encoding="utf-8") as f:
        file_data: list = json.load(f)

    if any(entry.get("name") == task_name for entry in file_data):
        print("Config entry already exists")
    else:
        new_config_entry = {
            "name": task_name,
            "task_type": "classification",
            "vocab": ["0", "1"],
            "trainset": name_train_fp,
            "validset": name_valid_fp,
            "testset": name_test_fp
        }
        file_data.append(new_config_entry)
        with Path(configs_path).open("w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4)
        print("Config Entry added")

    
    # If no model already exists -> Train
    # If flag is set -> Train
    # otherwise skip to prediction
    if not Path(model_path).exists() or do_train == True:
        print(f"Start Finetuning for task: {task_name}")
        shutil.copyfile(configs_path, 'configs.json')
        cmd = [
            "python",
            f"./models/{lm}/train_ditto.py",
            "--task", task_name,
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
        print(f"Finished Finetuning for task: {task_name}")

    config, model = load_model(task_name, "./models/ditto/checkpoints",
                            "roberta", True, False)
    model.eval()

    predict(name_test_fp,f"./ditto_out/{task_name}.jsonl", config, model,
            summarizer=None,
            max_len=128,
            lm="roberta",
            dk_injector=None,
            threshold=None)
    
    acc, prec, rec, f1 = calc_metrics(f"./ditto_out/{task_name}.jsonl", name_test_fp)
    print(task_name, acc, prec, rec, f1)

    # FINE TUNE MOVIE TABLE WITH REL SCORE
    task_movie_rel = f"{dataset}_movies_rel_score"
    lm = "ditto"
    do_train = True

    movie_train_fp = f"{movie_dir}/train_rel_score.txt"
    movie_valid_fp = f"{movie_dir}/valid_rel_score.txt"
    movie_test_fp = f"{movie_dir}/test_rel_score.txt"

    checkpoint_dir = f"./models/ditto/checkpoints/{task_movie_rel}"
    model_path = f"./models/ditto/checkpoints/{task_movie_rel}/model.pt"
    configs_path = f"./models/ditto/configs.json"
    log_dir = f"./logs"
    index_column = "id"

    with Path(configs_path).open("r", encoding="utf-8") as f:
        file_data: list = json.load(f)

    if any(entry.get("name") == task_movie_rel for entry in file_data):
        print("Config entry already exists")
    else:
        new_config_entry = {
            "name": task_movie_rel,
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
        print(f"Start Finetuning for task: {task_movie_rel}")
        shutil.copyfile(configs_path, 'configs.json')
        cmd = [
            "python",
            f"./models/{lm}/train_ditto.py",
            "--task", task_movie_rel,
            "--batch_size", "32",
            "--max_len", "128",
            "--lr", "3e-5",
            "--n_epochs", "1",
            "--finetuning",
            "--lm", "roberta",
            "--checkpoint_path", f"./models/ditto/checkpoints/{task_movie}/model.pt",
            "--fp16",
            "--save_model",
            "--logdir", "./models/ditto/checkpoints/",
        ]

        env = os.environ.copy()
        #env["CUDA_VISIBLE_DEVICES"] = "0"

        subprocess.run(cmd, env=env)
        print(f"Finished Finetuning for task: {task_movie_rel}")

    config, model = load_model(task_movie_rel, "./models/ditto/checkpoints",
                            "roberta", True, False)
    model.eval()

    predict(movie_test_fp,f"./ditto_out/{task_movie_rel}.jsonl", config, model,
            summarizer=None,
            max_len=128,
            lm="roberta",
            dk_injector=None,
            threshold=None)
    
    acc, prec, rec, f1 = calc_metrics(f"./ditto_out/{task_movie_rel}.jsonl", movie_test_fp)
    print(task_movie_rel, acc, prec, rec, f1)

    # FINE TUNE NAME TABLE

    task_name_rel = f"{dataset}_names_rel_score"
    lm = "ditto"
    do_train = True


    name_train_fp = f"{name_dir}/train_rel_score.txt"
    name_valid_fp = f"{name_dir}/valid_rel_score.txt"
    name_test_fp = f"{name_dir}/test_rel_score.txt"
    model_path = f"./models/ditto/checkpoints/{task_name_rel}/model.pt"
    configs_path = f"./models/ditto/configs.json"


    with Path(configs_path).open("r", encoding="utf-8") as f:
        file_data: list = json.load(f)

    if any(entry.get("name") == task_name_rel for entry in file_data):
        print("Config entry already exists")
    else:
        new_config_entry = {
            "name": task_name_rel,
            "task_type": "classification",
            "vocab": ["0", "1"],
            "trainset": name_train_fp,
            "validset": name_valid_fp,
            "testset": name_test_fp
        }
        file_data.append(new_config_entry)
        with Path(configs_path).open("w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4)
        print("Config Entry added")

    # If no model already exists -> Train
    # If flag is set -> Train
    # otherwise skip to prediction
    if not Path(model_path).exists() or do_train == True:
        print(f"Start Finetuning for task: {task_name_rel}")
        shutil.copyfile(configs_path, 'configs.json')
        cmd = [
            "python",
            f"./models/{lm}/train_ditto.py",
            "--task", task_name_rel,
            "--batch_size", "32",
            "--max_len", "128",
            "--lr", "3e-5",
            "--n_epochs", "1",
            "--finetuning",
            "--lm", "roberta",
            "--checkpoint_path", f"./models/ditto/checkpoints/{task_name}/model.pt",
            "--fp16",
            "--save_model",
            "--logdir", "./models/ditto/checkpoints/",
        ]

        env = os.environ.copy()
        #env["CUDA_VISIBLE_DEVICES"] = "0"

        subprocess.run(cmd, env=env)
        print(f"Finished Finetuning for task: {task_name_rel}")

    config, model = load_model(task_name_rel, "./models/ditto/checkpoints",
                            "roberta", True, False)
    model.eval()

    predict(name_test_fp,f"./ditto_out/{task_name_rel}.jsonl", config, model,
            summarizer=None,
            max_len=128,
            lm="roberta",
            dk_injector=None,
            threshold=None)
    
    acc, prec, rec, f1 = calc_metrics(f"./ditto_out/{task_name_rel}.jsonl", name_test_fp)
    print(task_name_rel, acc, prec, rec, f1)

if __name__=="__main__":
    main()