# config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional dependency: pyyaml
try:
    import yaml
except ImportError as e:
    raise RuntimeError("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from e


# ----------------------------
# Typed configs
# ----------------------------

@dataclass(frozen=True)
class DatasetConfig:
    dataset_dir: Path
    splits: Tuple[str, ...] = ("train", "valid", "test")
    split_ext: str = "csv"

    # pairs schema
    left_id_col: str = "ltable_id"
    right_id_col: str = "rtable_id"
    label_col: str = "label"

    # primary/context schema
    prim_id_col: str = "id"
    context_field: str = "authors"
    context_sep: str = ","
    context_name_field: str = "name"

    # how to map pairs->tables (deepmatcher: tableA/tableB)
    table_configs: Tuple[Tuple[str, str], ...] = (("a", "ltable_id"), ("b", "rtable_id"))


@dataclass(frozen=True)
class RunConfig:
    run_name: str

    # roots
    output_root: Path = Path("./data/processed")
    reduced_root: Path = Path("./data/processed/reduced")
    logs_root: Path = Path("./logs")
    checkpoints_root: Path = Path("./models/ditto/checkpoints")
    ditto_configs_path: Path = Path("./models/ditto/configs.json")

    # naming/layout knobs
    splits_dirname: str = "splits"   # output_root/<splits_dirname>/<run_name>
    model_filename: str = "model.pt"


@dataclass(frozen=True)
class FinetuneConfig:
    enabled: bool = True
    task: str = "testing-setup"   # ditto task name
    lm: str = "ditto"             # which training script folder to use (models/<lm>/...)
    encoder: str = "roberta"      # ditto --lm argument (roberta/distilbert/etc.)

    batch_size: int = 32
    max_len: int = 128
    lr: float = 3e-5
    n_epochs: int = 1
    fp16: bool = True


@dataclass(frozen=True)
class AlgoConfig:
    # keep only algorithm knobs here
    alpha: float = 0.1
    threshold: float = 0.5
    k: int = 2

    # strategy/sim choices
    strategy: str = "LATE_FUSION"           # string so YAML is simple
    rel_similarity: str = "jaccard_coefficient"
    is_relations_triples: bool = True

    # runtime
    use_gpu: bool = True
    fp16: bool = True


@dataclass(frozen=True)
class PathsConfig:
    """
    Derived, fully-resolved paths used by the pipeline.
    """
    dataset_dir: Path
    output_dir: Path
    reduced_dir: Path
    log_dir: Path
    checkpoint_dir: Path
    model_path: Path
    ditto_configs_path: Path


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    run: RunConfig
    algo: AlgoConfig
    finetune: FinetuneConfig
    paths: PathsConfig


# ----------------------------
# YAML loading + normalization
# ----------------------------

def _as_path(p: Any) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def _coerce_paths_in_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    out = dict(d)
    for k in keys:
        if k in out and out[k] is not None:
            out[k] = _as_path(out[k])
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must be a mapping at top-level, got: {type(data)}")
    return data


def _require_sections(data: Dict[str, Any], required: Tuple[str, ...]) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}. Present: {list(data.keys())}")


def _derive_paths(dataset: DatasetConfig, run: RunConfig, finetune: FinetuneConfig) -> PathsConfig:
    output_dir = run.output_root / run.splits_dirname / run.run_name
    checkpoint_dir = run.checkpoints_root / finetune.task
    model_path = checkpoint_dir / run.model_filename

    return PathsConfig(
        dataset_dir=dataset.dataset_dir,
        output_dir=output_dir,
        reduced_dir=run.reduced_root,
        log_dir=run.logs_root,
        checkpoint_dir=checkpoint_dir,
        model_path=model_path,
        ditto_configs_path=run.ditto_configs_path,
    )


def load_config(yaml_path: str | Path) -> AppConfig:
    """
    Single entrypoint used by your runner. All YAML reading/validation happens here.
    """
    path = _as_path(yaml_path)
    data = _read_yaml(path)
    _require_sections(data, required=("dataset", "run", "algo"))

    # dataset
    dataset_raw = _coerce_paths_in_dict(data["dataset"], ["dataset_dir"])
    # yaml lists -> tuples
    if "splits" in dataset_raw and isinstance(dataset_raw["splits"], list):
        dataset_raw["splits"] = tuple(dataset_raw["splits"])
    if "table_configs" in dataset_raw and isinstance(dataset_raw["table_configs"], list):
        dataset_raw["table_configs"] = tuple(tuple(x) for x in dataset_raw["table_configs"])

    dataset = DatasetConfig(**dataset_raw)

    # run
    run_raw = _coerce_paths_in_dict(
        data["run"],
        ["output_root", "reduced_root", "logs_root", "checkpoints_root", "ditto_configs_path"],
    )
    run = RunConfig(**run_raw)

    # algo
    algo = AlgoConfig(**data["algo"])

    # finetune (optional)
    finetune_raw = data.get("finetune", {})
    finetune = FinetuneConfig(**finetune_raw)

    # derived paths
    paths = _derive_paths(dataset, run, finetune)

    return AppConfig(dataset=dataset, run=run, algo=algo, finetune=finetune, paths=paths)


# ----------------------------
# Optional: convenience adapter-like helpers (still config-only)
# ----------------------------

def split_pairs_path(dataset: DatasetConfig, split: str) -> Path:
    return dataset.dataset_dir / f"{split}.{dataset.split_ext}"


def table_path(dataset: DatasetConfig, suffix: str) -> Path:
    return dataset.dataset_dir / f"table{suffix.upper()}.csv"
