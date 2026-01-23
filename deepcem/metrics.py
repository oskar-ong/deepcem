import logging
from typing import List, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from deepcem.data_structures import get_cluster

def get_pred_labels(y_true, candidates, references, clusters, parents, index_column):
    result_set = []
    for _, row in candidates.iterrows():
        left_id = row[0][index_column]
        right_id = row[1][index_column]
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