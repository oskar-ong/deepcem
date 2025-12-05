import logging
from typing import List, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from deepcem.data_structures import get_cluster

def classification_metrics(
    y_true: List[Union[int, str]],
    y_pred: List[Union[int, str]],
    average: str = "binary"
) -> dict:
    """
    Calculate accuracy, precision, recall, and F1-score.

    Parameters
    ----------
    y_true : list
        List of expected (ground truth) labels.
    y_pred : list
        List of predicted labels.
    average : str, optional
        Averaging method for multi-class or multi-label tasks.
        Options: 'binary', 'micro', 'macro', 'weighted'. Default is 'binary'.

    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, precision, recall, and F1-score.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics

def get_pred_labels(y_true, candidates, references, clusters, parents):
    result_set = []
    for _, row in candidates.iterrows():
        left_id = row[0]["Id"]
        right_id = row[1]["Id"]
        left_cluster = get_cluster(
            references[left_id].cluster, clusters, parents)
        right_cluster = get_cluster(
            references[right_id].cluster, clusters, parents)
        label = 1 if left_cluster == right_cluster else 0
        result_set.append(label)

        logger = logging.getLogger('cem')
        logger.info(
            f"label: {label} | | left_id: {left_id} - left_cluster:{left_cluster} | right_id: {right_id} - right cluster: {right_cluster}")

    y_pred = result_set

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1

def get_true_labels(cfg):
    with open(f"data/interim/{cfg.dataset}/extracted/splits/test.txt", "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    y_true = [int(l.split("\t")[2]) for l in lines]
    return y_true

if __name__ == "__main__":
    print('not implemented yet')
