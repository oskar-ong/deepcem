import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Tuple
import uuid

from evaluate import calc_metrics
from experiment_config import DITTO_CONFIG


def get_unique_config(task, train_suffix, dataset_dir, special_tokens):

    if train_suffix == "":
        train_fp = f"{dataset_dir}/train.txt"
    else:
        train_fp = f"{dataset_dir}/train_{train_suffix}.txt"
    valid_fp = f"{dataset_dir}/valid.txt"
    test_fp = f"{dataset_dir}/test.txt"

    new_config_entry = {
        "name": task,
        "task_type": "classification",
        "vocab": ["0", "1"],
        "trainset": train_fp,
        "validset": valid_fp,
        "testset": test_fp,
        "tokens": special_tokens
    }

    with open(f"./models/ditto/configs.json", 'r') as f:
        full_configs = json.load(f)

    full_configs.append(new_config_entry)

    temp_config_path = f"config_{task}_{uuid.uuid4().hex[:8]}.json"

    with open(temp_config_path, 'w') as f:
        json.dump(full_configs, f)

    return temp_config_path


def finetune(task, dataset_dir, log, special_tokens, train_suffix="", seed=0):

    config_path = get_unique_config(
        task, train_suffix, dataset_dir, special_tokens)

    # ================================================================================
    # --- Finetune ---
    # ================================================================================
    cmd = [
        "python",
        f"./models/ditto/train_ditto.py",
        "--task", task,
        "--batch_size", f"{DITTO_CONFIG['batch_size']}",
        "--max_len", f"{DITTO_CONFIG['max_len']}",
        "--lr", f"{DITTO_CONFIG['learning_rate']}",
        "--n_epochs", f"{DITTO_CONFIG['epochs']}",
        "--finetuning",
        "--lm", f"{DITTO_CONFIG['lm']}",
        "--fp16",
        "--save_model",
        "--logdir", "./models/ditto/checkpoints/",
        "--run_id", f"{seed}",
        "--config_path", config_path,
    ]

    env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        log.error(f"Finetune phase 1 failed for task {task}. Error: {e}")
        raise  # Re-raise to alert Slurm
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)
    log.info(f"Finished Finetuning for task: {task}")


def refinetune(task, dataset_dir, base, log, special_tokens, train_suffix="", seed=0):

    configs_path = get_unique_config(
        task, train_suffix, dataset_dir, special_tokens)

    # ================================================================================
    # --- Finetune ---
    # ================================================================================
    cmd = [
        "python",
        f"./models/ditto/train_ditto.py",
        "--task", task,
        "--batch_size", f"{DITTO_CONFIG['batch_size']}",
        "--max_len", f"{DITTO_CONFIG['max_len']}",
        "--lr", f"{DITTO_CONFIG['learning_rate']}",
        "--n_epochs", f"{DITTO_CONFIG['epochs']}",
        "--finetuning",
        "--lm", f"{DITTO_CONFIG['lm']}",
        "--checkpoint_path", f"./models/ditto/checkpoints/{base}/model.pt",
        "--fp16",
        "--save_model",
        "--logdir", "./models/ditto/checkpoints/",
        "--run_id", f"{seed}",
        "--config_path", configs_path,
    ]

    env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        log.error(f"Finetune phase 2 failed for task {task}. Error: {e}")
        raise  # Re-raise to alert Slurm
    finally:
        if os.path.exists(configs_path):
            os.remove(configs_path)
    log.info(f"Finished Finetuning for task: {task}")


def evaluate(task, input_path, output_path, log, true_fp: str, dataset_dir, special_tokens=None) -> Tuple[float, float, float, float]:

    log.info("Start Predicting: ")
    log.info(f"Task: {task} ")
    log.info(f"Output Path: {output_path} ")
    log.info(f"Input Path: {input_path} ")

    if special_tokens is None:
        special_tokens = []

    config_path = get_unique_config(
        task, "", dataset_dir, special_tokens)

    cmd = [
        "python",
        f"./models/ditto/matcher.py",
        "--task", task,
        "--input_path", input_path,
        "--output_path", output_path,
        "--lm", f"{DITTO_CONFIG['lm']}",
        "--max_len", f"{DITTO_CONFIG['max_len']}",
        "--use_gpu",
        "--fp16",
        "--checkpoint_path", "./models/ditto/checkpoints/",
        "--config_path", config_path,
    ]

    env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        log.error(f"Prediction failed for task {task}. Error: {e}")
        raise  # Re-raise to alert Slurm
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

    log.info(f"Finished Predicting")

    acc, prec, rec, f1 = calc_metrics(output_path, true_fp)
    acc, prec, rec, f1 = round(acc, 3), round(
        prec, 3), round(rec, 3), round(f1, 3)
    log.info(f"{task} METRICS: Accuracy, Prediction, Recall, F1")
    log.info(f"{task} METRICS: {acc}, {prec}, {rec}, {f1}")
    print(f"{task} {input_path} {output_path} METRICS", acc, prec, rec, f1)
    return acc, prec, rec, f1
