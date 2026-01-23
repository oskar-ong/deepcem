import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from deepcem.clustering import UnionFind, bootstrap_clusters, init_pq, iterative_merge
from deepcem.config import AppConfig, load_config, split_pairs_path, table_path
from deepcem.graph import extract_references
from deepcem.metrics import get_pred_labels
from deepcem.preprocessing import (
    build_block_index,
    enrich_pairs_dataframe,
    match_context_entities,
    normalize,
    normalize_simple_context,
    reduce,
    save_as_ditto_file,
)
from deepcem.utils import attach_run_file_handler, setup_base_logger

@dataclass
class SideData:
    primary_entities: Dict[str, Any]
    context_entities: Dict[str, Any]
    primary_to_context: Dict[str, List[str]]
    context_to_primary: Dict[str, List[str]]

@dataclass
class SplitArtifacts:
    pairs: pd.DataFrame
    enriched_pairs: pd.DataFrame
    # For each side (a/b): dicts from normalize()
    # { "primary_entities": ..., "context_entities": ..., "primary_to_contexxt": ..., "context_to_primary": ... }
    side_data: Dict[str, SideData]


# ----------------------------
# Stage 1: preprocess splits
# ----------------------------

def preprocess_split(
    config: AppConfig,
    split: str,
) -> SplitArtifacts:
    """
    For a split:
      - reduce tableA/tableB w.r.t. pairs
      - normalize reduced tables into primary_entities/context_entities/edges
      - enrich pairs into Ditto format and save
    """
    pairs_fp = split_pairs_path(config.dataset, split)
    pairs_df = pd.read_csv( pairs_fp,
                            dtype={config.dataset.left_id_col: str, config.dataset.right_id_col: str},
                            )

    side_data: Dict[str, SideData] = {}

    for suffix, id_col_in_pairs in config.dataset.table_configs:
        reduced_fp = config.paths.reduced_dir / f"{split}_{suffix}.csv"
        table_raw_fp = table_path(config.dataset, suffix)

        # Reduce: keep only records referenced by this split
        reduce(
            str(pairs_fp),
            str(table_raw_fp),
            str(reduced_fp),
            config.dataset.prim_id_col,
            id_col_in_pairs,
        )

        # Normalize reduced table (expects columns/logic inside normalize())
        norm_primary_entities, norm_context_entities, prim_to_context, context_to_prim = normalize(
            str(reduced_fp),
            config.dataset.prim_id_col,
            config.dataset.context_field,
            config.dataset.context_sep,
        )

        side_data[suffix] = SideData(
            primary_entities=norm_primary_entities,
            context_entities=norm_context_entities,
            primary_to_context=prim_to_context,
            context_to_primary=context_to_prim
        )

    enriched = enrich_pairs_dataframe(
        pairs_df,
        side_data["a"].primary_entities,
        side_data["b"].primary_entities,
    )

    # Save as Ditto file
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    save_as_ditto_file(enriched, str(config.paths.output_dir), split)

    return SplitArtifacts(
        pairs=pairs_df,
        enriched_pairs=enriched,
        side_data=side_data,
    )


def preprocess_all_splits(
    config: AppConfig,
) -> Dict[str, SplitArtifacts]:
    logger = logging.getLogger("cem")
    config.paths.reduced_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, SplitArtifacts] = {}
    for split in config.dataset.splits:
        artifacts[split] = preprocess_split(config, split)
    return artifacts


# ----------------------------
# Stage 2: optional finetune
# ----------------------------

def ensure_ditto_task_config(
    task: str,
    configs_path: Path,
    output_dir: Path,
) -> None:
    configs_path.parent.mkdir(parents=True, exist_ok=True)
    if configs_path.exists():
        with configs_path.open("r", encoding="utf-8") as f:
            file_data: list = json.load(f)
    else:
        file_data = []

    if any(entry.get("name") == task for entry in file_data):
        return

    file_data.append(
        {
            "name": task,
            "task_type": "classification",
            "vocab": ["0", "1"],
            "trainset": str(output_dir / "train.txt"),
            "validset": str(output_dir / "valid.txt"),
            "testset": str(output_dir / "test.txt"),
        }
    )
    with configs_path.open("w", encoding="utf-8") as f:
        json.dump(file_data, f, indent=4)


def finetune_ditto_if_needed(
    config: AppConfig,
    batch_size: int = 32,
    max_len: int = 128,
    lr: str = "3e-5",
    n_epochs: str = "1",
) -> None:
    logger = logging.getLogger("cem")
    ensure_ditto_task_config(config.finetune.task, config.paths.ditto_configs_path, config.paths.output_dir)

    if config.paths.model_path.exists():
        logger.info("[finetune] model exists, skipping: %s", config.paths.model_path)
        return

    logger.info("[finetune] model missing, fine-tuning…")
    config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Some Ditto scripts look for configs.json in cwd
    shutil.copyfile(config.paths.ditto_configs_path, Path("configs.json"))

    cmd = [
        "python",
        f"./models/ditto/train_ditto.py",
        "--task", config.finetune.task,
        "--batch_size", str(batch_size),
        "--max_len", str(max_len),
        "--lr", str(lr),
        "--n_epochs", str(n_epochs),
        "--finetuning",
        "--lm", "roberta",
        "--fp16",
        "--save_model",
        "--logdir", "./models/ditto/checkpoints/",
    ]
    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=False)


# ----------------------------
# Stage 3: build context graph (test-time)
# ----------------------------

def build_context_clusters_and_hyperedges(
    artifacts_test: SplitArtifacts, config: AppConfig,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Returns:
      - primary_to_context: dict[primary_id, list[context_ent_cluster_id]]
      - context_to_primary: dict[context_cluster_id, list[primary_id]]
    """
    logger = logging.getLogger("cem")
    # Merge mentions (context_entities) across sides
    all_context_entities: dict = artifacts_test.side_data["a"].context_entities | artifacts_test.side_data["b"].context_entities

    # Normalize names for blocking/matching
    for k in all_context_entities.keys():
        all_context_entities[k]["normalized"] = normalize_simple_context(all_context_entities[k][config.dataset.context_name_field])

    blocks = build_block_index(all_context_entities)

    uf = UnionFind()
    for entity in all_context_entities:
        uf.add(entity)

    logger.info("[context_entities] before matching: %s", len(uf.get_sets()) )
    uf = match_context_entities(all_context_entities, blocks, uf)
    logger.info("[context_entities] after matching : %s", len(uf.get_sets()) )

    mention_to_cluster = {m: uf.find(m) for m in all_context_entities}

    # primary -> context mentions
    primary_to_context_raw = artifacts_test.side_data["a"].primary_to_context | artifacts_test.side_data["b"].primary_to_context

    # primary -> context clusters
    primary_to_context: Dict[str, List[str]] = {}
    for prim, mentions in primary_to_context_raw.items():
        clusters = {mention_to_cluster[m] for m in mentions if m in mention_to_cluster}
        primary_to_context[prim] = list(clusters)

    # context cluster -> primary
    context_to_primary: Dict[str, List[str]] = {}
    for prim_id, clusters in primary_to_context.items():
        for c in clusters:
            context_to_primary.setdefault(c, []).append(prim_id)

    return primary_to_context, context_to_primary


# ----------------------------
# Stage 4: run CEM + evaluate
# ----------------------------

def run_cem(
    enriched_pairs_test: pd.DataFrame,
    pairs_test: pd.DataFrame,
    primary_to_context: Dict[str, List[str]],
    context_to_primary: Dict[str, List[str]],
    config: AppConfig
) -> Tuple[float, float, float, float]:
    cs = pd.DataFrame(enriched_pairs_test)
    references = extract_references(cs)

    hyperedges = primary_to_context

    clusters, parents = bootstrap_clusters(cs, references, hyperedges, context_to_primary)
    pq, clusters = init_pq(clusters, references, hyperedges, parents, config.algo)

    clusters, references, hyperedges, parents = iterative_merge(
        pq, clusters, parents, hyperedges, references, config.algo
    )

    y_true = pairs_test[config.dataset.label_col].astype(int).tolist()
    acc, prec, rec, f1 = get_pred_labels(
        y_true, cs, references, clusters, parents, config.dataset.prim_id_col
    )
    return acc, prec, rec, f1


# ----------------------------
# CLI + main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    logger = setup_base_logger("cem")
    attach_run_file_handler(logger, str(config.paths.log_dir), config.algo.alpha, config.algo.threshold)

    # 1) preprocess all splits (dataset-agnostic pipeline)
    artifacts = preprocess_all_splits(config)

    # 2) optional finetuning
    if config.finetune.enabled:
        finetune_ditto_if_needed(config)

    # 3) build test-time context graph
    primary_to_context, context_to_primary = build_context_clusters_and_hyperedges(artifacts["test"], config)

    # 4) run CEM + evaluate
    acc, prec, rec, f1 = run_cem(
        enriched_pairs_test=artifacts["test"].enriched_pairs,
        pairs_test=artifacts["test"].pairs,
        primary_to_context=primary_to_context,
        context_to_primary=context_to_primary,
        config=config
    )

    logger.info(f"threshold : {config.algo.threshold}")
    logger.info(f"alpha : {config.algo.alpha}")
    logger.info(f"accuracy : {round(acc, 2)}")
    logger.info(f"precision: {round(prec,2)}")
    logger.info(f"recall   : {round(rec,2)}")
    logger.info(f"f1       : {round(f1,2)}")
    logger.info("=== End Pipeline Run ===")

if __name__ == "__main__":
    main()
