from dataclasses import dataclass, replace
from pathlib import Path
import json

LATE_FUSION = 1
INLINE_ENCODE = 2

@dataclass
class PipelineConfig:

    dataset = "citations_extracted"
    alpha = 0.1
    threshold = 0.5
    k = 2
    ckpt = "models/ditto/checkpoints"
    lm = "roberta"
    use_gpu = False
    fp16 = False
    is_relations_triples = True
    rel_similarity = "jaccard_coefficient"
    strategy = LATE_FUSION
    log_dir = "notebooks/output/logs"
    
    # --------- loading from file (json / txt / .config) ---------
    @classmethod
    def from_file(cls, path: str) -> "PipelineConfig":
        """
        Load config values from a json-like file stored as a dict.
        Example file content:
            {"threshold": 0.7, "batch_size": 64}
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    # --------- runtime overrides (immutably) ---------
    def with_overrides(self, **overrides) -> "PipelineConfig":
        """
        Return a *new* config with some fields changed.
        """
        return replace(self, **overrides)
