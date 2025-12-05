import re
from typing import Dict, List, Tuple
from deepcem.data_structures import Cluster, Hyperedge, Reference
from deepcem.similarity import cluster_stats, pairwise_neighbor_stats

def sanitize_value(v: str) -> str:
    if v is None:
        return ""
    v = str(v)

    if v == "nan":
        return ""

    # Remove tabs, convert to space
    v = v.replace("\t", " ")

    # Remove newlines
    v = v.replace("\n", " ").replace("\r", " ")

    # Collapse multiple spaces
    v = re.sub(r"\s+", " ", v).strip()

    return v

def serialize_record_base(ref: Reference):
    return " ".join(f"COL {k} VAL {sanitize_value(v)}" for k, v in ref.attrs.items())

def serialize_record_with_typed_neighbors_and_stats(
    ref: Reference,
    cluster: Cluster,
    hyperedges: Dict[str, Hyperedge],
    references: Dict[str, Reference],
) -> str:
    result = serialize_record_base(ref)

    # 2) stats → numeric attributes (still text)
    stats = cluster_stats(cluster, hyperedges, references)
    for name, value in stats.items():
        result += f" COL {name} VAL {value}"

    return result


def serialize_pair(
    left_ref: Reference,
    left_cluster: Cluster,
    right_ref: Reference,
    right_cluster: Cluster,
    hyperedges: Dict[str, Hyperedge],
    references: Dict[str, Reference],
    label: int,
) -> str:
    # left / right with typed neighbors + cluster stats
    left_entry = serialize_record_with_typed_neighbors_and_stats(
        left_ref, left_cluster, hyperedges, references
    )
    right_entry = serialize_record_with_typed_neighbors_and_stats(
        right_ref, right_cluster, hyperedges, references
    )

    # pairwise stats
    pw_stats = pairwise_neighbor_stats(
        left_cluster, right_cluster, hyperedges, references
    )

    # inject into both left and right entries
    # (you can also only add to left if you want, but symmetry is okay)
    for name, value in pw_stats.items():
        left_entry += f" COL pair_{name} VAL {value}"
        right_entry += f" COL pair_{name} VAL {value}"

    return f"{left_entry}\t{right_entry}\t{label}"


def serialize_to_ditto_wo_id(
    data: List[Tuple[Dict[str, str], Dict[str, str], int]]
) -> List[str]:
    
    def to_entry(d: Dict[str, str]) -> str:
        return " ".join(
            f"COL {k} VAL {sanitize_value(v)}"
            for k, v in d.items()
            if k != "id"
        )

    return [
        f"{to_entry(left)}\t{to_entry(right)}\t{label}"
        for left, right, label in data
    ]