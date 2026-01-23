import re
from typing import Dict, List, Tuple

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