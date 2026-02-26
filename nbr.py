import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from deepcem.serialize import serialize_to_ditto_wo_id
from deepcem.utils import get_attrs_for_keys
from matcher import load_model, classify 

dataset = "Dn7"
task = f"{dataset}_nbr_encoded"
lm = "ditto"
dataset_dir = f"data/raw/{dataset}"
output_dir = f"./data/processed/{lm}-splits/{dataset}"
checkpoint_dir = f"./models/ditto/checkpoints/{task}"
model_path = f"./models/ditto/checkpoints/{task}/model.pt"
configs_path = f"./models/ditto/configs.json"
log_dir = f"./logs"
index_column = "id"
context_column = "brand"
encoding_variant = 1

def main():
    task = task + str(encoding_variant)

    train_fp = f"{dataset_dir}/train.csv" 
    train_df = pd.read_csv(train_fp)

    valid_df = pd.read_csv(f"{dataset_dir}/valid.csv")

    test_df = pd.read_csv(f"{dataset_dir}/test.csv")

    df_A = pd.read_csv(f"{dataset_dir}/tableA.csv")  # left

    df_B = pd.read_csv(f"{dataset_dir}/tableB.csv")  # right

    ids_A_train = set(train_df["left_id"])
    ids_B_train = set(train_df["right_id"])

    df_A_filtered_train = df_A[df_A["id"].isin(ids_A_train)]
    df_B_filtered_train = df_B[df_B["id"].isin(ids_B_train)]

    def create_nbr(df: pd.DataFrame) -> dict:
        df2 = df.copy()
        df2[context_column] = (
            df2[context_column]
                .astype(str)
                .str.strip()
                .replace(r'^"+$', "", regex=True)   # remove strings made only of quotes
                .replace("", pd.NA)
        )
        m = df2[context_column].astype("string").str.strip()
        df2[context_column] = m.mask(m.eq(""), pd.NA)


        manuf_to_products = (
            df2.dropna(subset=[context_column])
                .groupby(context_column)["title"]
                .apply(list)
        )

        df2["Neighborhood"] = df2[context_column].map(manuf_to_products).apply(
            lambda x: x if isinstance(x, list) else []
        )

        df2["Neighborhood"] = df2.apply(
            lambda r: [p for p in r["Neighborhood"] if p != r["title"]],
            axis=1
        )

        id_to_nbr = df2.set_index("id")["Neighborhood"].to_dict()
        return id_to_nbr

    nbr1 = create_nbr(df_A_filtered_train)
    nbr2 = create_nbr(df_B_filtered_train)

    print(nbr1[387])
    print(nbr2[3169])

    # create ditto files
    if encoding_variant == 1:
        encode_as_string()
    if encoding_variant == 2: 
        encode_as_float()
    if encoding_variant == 3: 
        encode_as_bucket()

    ## 3) same as 2) but as buckets low, medium, high

    # Convert to string once and index by 'id'
    df_A_idx = df_A.set_index(index_column)
    df_B_idx = df_B.set_index(index_column)

    train = get_attrs_for_keys(df_A_idx, df_B_idx, train_df)
    valid = get_attrs_for_keys(df_A_idx, df_B_idx, valid_df)
    test  = get_attrs_for_keys(df_A_idx, df_B_idx, test_df)

    train_ditto = serialize_to_ditto_wo_id(train)
    with open(f"{output_dir}/base/train.txt","w", encoding="utf-8") as f:
        f.write("\n".join(train_ditto))

    valid_ditto = serialize_to_ditto_wo_id(valid)
    with open(f"{output_dir}/base/valid.txt","w", encoding="utf-8") as f:
        f.write("\n".join(valid_ditto))
        
    test_ditto = serialize_to_ditto_wo_id(test)
    with open(f"{output_dir}/base/test.txt","w", encoding="utf-8") as f:
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
            "trainset": f"{output_dir}/base/train.txt",
            "validset": f"{output_dir}/base/valid.txt",
            "testset": f"{output_dir}/base/test.txt"
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

    scores = classify(test_ditto, model, lm="roberta", max_len=128)

    y_true = [int(l[2]) for l in test]

    y_pred = scores[0] 

    base_acc = accuracy_score(y_true, y_pred)
    base_prec = precision_score(y_true, y_pred, zero_division=0)
    base_rec = recall_score(y_true, y_pred, zero_division=0)
    base_f1 = f1_score(y_true, y_pred, zero_division=0)
    print("accuracy :", base_acc)
    print("precision:", base_prec)
    print("recall   :", base_rec)
    print("f1       :", base_f1)

def encode_as_string():
    ## 1) neighborhoods as string block


    ## 1a) shortened at cut off length (top k)

    ## 1b) ditto summarize

    ## 1c) own summarizer?

    ## 1d) product ids instead of title

    pass 
def encode_as_float():
    ## 2) similarity metric
    ## a) which metrics? 
    pass 
def encode_as_bucket():
    ## 3) same as 2) but as buckets low, medium, high
    pass 

if __name__=="__main__":
    main()