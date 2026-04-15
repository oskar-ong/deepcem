import argparse
from pathlib import Path
import pickle
import shutil
from typing import Dict

from entity_config import REGISTRY, EntityConfig
from serializer import write_splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--bin", action="store_true")
    args = parser.parse_args()
    CONFIGS: Dict[str, EntityConfig] = REGISTRY[args.dataset]

    with open(f"pickles/{args.dataset}_processed_entities.pickle", 'rb') as f:
        processed_entities = pickle.load(f)

    # print(processed_entities)

    with open(f"pickles/{args.dataset}_relation_maps.pickle", 'rb') as f:
        relation_maps = pickle.load(f)

    # --- Write Splits to Disk ---
    for cfg_name, cfg in CONFIGS.items():
        # Write splits for each pollution level
        for level, df in processed_entities[cfg.name].dfs_by_pollution.items():
            write_splits(cfg, CONFIGS, processed_entities,
                         relation_maps, level, True)

    copied = set()
    for cfg in CONFIGS.values():
        for relation in cfg.rels:
            # Copy junction table to new dir
            if not cfg.path_out_dir in copied:

                file_name = Path(relation['junction_table']).name

                shutil.copyfile(
                    relation["junction_table"], f"{cfg.path_out_dir}/{file_name}")
            copied.add(f"{cfg.path_out_dir}/{file_name}")
