import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calc_metrics(predict_fp, true_fp):
    # Initialize local variables as lists
    y_pred = []
    y_true = []

    with open(predict_fp, "r", encoding="utf-8") as f:
        for line in f:
            # Parse the JSON string into a dictionary
            data = json.loads(line)
            
            # Access and store the specific attributes
            y_pred.append(data["match"])

    with open(true_fp, "r", encoding="utf-8") as f:
        for line in f:
            # .strip() removes trailing newlines (\n)
            # [-1] grabs the last character remaining
            last_char = line.strip()[-1]
            y_true.append(int(last_char))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1

    