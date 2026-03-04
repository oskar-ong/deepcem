import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calc_metrics(predict_fp, true_fp: str):
    y_pred = []
    y_true = []
    true_path = Path(true_fp)

    with open(predict_fp, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            y_pred.append(int(data["match"]))

    with open(true_fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            

            if true_path.suffix.lower() == '.jsonl':
                data_list = json.loads(line)
                label = data_list[2] # Accessing the 3rd object
                y_true.append(int(label))

            elif true_path.suffix.lower() == '.txt':
                # Grabs the last character remaining
                last_char = line[-1]
                y_true.append(int(last_char))


    # Safety Check: Ensure the files matched up
    if len(y_pred) != len(y_true):
        raise ValueError(f"Data mismatch! Found {len(y_pred)} predictions but {len(y_true)} ground truth labels.")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return acc, prec, rec, f1