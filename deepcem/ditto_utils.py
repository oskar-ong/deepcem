from deepcem.data_structures import Reference
def serialize_ditto(ri: Reference, rj: Reference):
    """
    Convert two pandas Series (records) into Ditto input format.

    Args:
        record_a (pd.Series): A row from dataframe A.
        record_b (pd.Series): A row from dataframe B.

    Returns:
        str: A Ditto-formatted string.
    """
    def serialize_record(record):
        parts = []
        for col, val in record.items():
            if val is None:
                continue
            parts.append(f'{col} : "{val}"')
        return " , ".join(parts)

    left = serialize_record(ri.attrs)
    right = serialize_record(rj.attrs)

    return f"{left}\t{right}\t0"


def interpret_ditto_predictions(labels, scores):
    """
    Convert Ditto outputs into human-readable predictions with probabilities.

    Args:
        labels (list[int]): predicted labels from Ditto (0=no match, 1=match)
        scores (list[list[float]]): logits from Ditto for each example, shape=[num_examples, 2]

    Returns:
        list of dict: each dict contains:
            - 'predicted_label': 0 or 1
            - 'predicted_class': "no match" or "match"
            - 'probabilities': [prob_no_match, prob_match]
    """
    results = []
    for label, logit in zip(labels, scores):
        # Compute probabilities via softmax
        probs = np.exp(logit) / np.sum(np.exp(logit))
        results.append({
            "predicted_label": label,
            "predicted_class": "match" if label == 1 else "no match",
            "probabilities": probs.tolist()
        })
    return results