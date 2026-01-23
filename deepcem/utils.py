import logging
import os

import pandas as pd

from deepcem.config import AlgoConfig

def setup_logging(fp):
    logger = logging.getLogger("cem")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(fp)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

import logging
from datetime import datetime
from pathlib import Path

def setup_base_logger(name="cem"):
    """Call this ONCE at program start."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # If running in notebooks / re-running cells:
    logger.handlers.clear()

    # Console handler (shared across all runs)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def attach_run_file_handler(logger: logging.Logger, log_dir: str, alpha, threshold):
    """Attach a run-specific file handler for this threshold."""
    # remove old FileHandlers (keep console handler)
    logger.handlers = [
        h for h in logger.handlers
        if not isinstance(h, logging.FileHandler)
    ]
    
    Path(f"{log_dir}").mkdir(exist_ok=True)
    threshold = str(threshold).replace(".", "_")
    alpha = str(alpha).replace(".", "-")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"{log_dir}/alpha_{alpha}_t_{threshold}_{ts}.log"

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging this run to {logfile}")

def get_row_as_dict_idx(df_idx: pd.DataFrame, key: str, value: str) -> dict[str, str]:
    """Return row from df_idx that matches id (index lookup, O(1))"""
    try:
        row = df_idx.loc[value]
    except KeyError:
        raise ValueError(f"No match found for ID='{value}' in table: {key}")
    result = row.to_dict()
    result['id'] = value
    return result

def get_attrs_for_keys(df_A_idx, df_B_idx, df_pairs: pd.DataFrame):
    result: list[tuple[dict, dict, int]] = []

    for row in df_pairs.itertuples(index=False):
        try:
            left = get_row_as_dict_idx(df_A_idx, "ltable", row.ltable_id)
            right = get_row_as_dict_idx(df_B_idx, "rtable", row.rtable_id)
            label = row.label
        except KeyError as e:
            raise KeyError(f"Missing expected key in row {row.Index if hasattr(row, 'Index') else '?'}: {e}")

        result.append((left, right, label))
    return result

def extract_relation(l, subject):
    relationships = []
    for row in l:
        
        author_left = row[0][subject]
        author_right = row[1][subject]

        create_rdf_triple(row, relationships, author_left, "isAuthor", ',', 'id')
        create_rdf_triple(row, relationships, author_right, "isAuthor", ',', 'id')

    return relationships

def create_rdf_triple(row, relationships, subject, predicate,delimiter, object):

    if not pd.isna(subject):

        subject = subject.split(delimiter)

        for r in subject:
            relationships.append((r.strip(), predicate, row[0][object]))

def set_log_dir(cfg: AlgoConfig):
    log_dir = f"{cfg.log_dir}/{cfg.dataset}"
    return log_dir

def get_run_id(log_dir):
    log_dirs = os.listdir(log_dir)
    log_dirs = [int(x) for x in log_dirs]

    if not log_dirs:
        run_id = 1
    else:
        latest_run_id = max(log_dirs)
        run_id = int(latest_run_id) + 1
    return run_id

def cluster_pair(a, b):
    return (a, b) if a <= b else (b, a)