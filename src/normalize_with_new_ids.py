import pandas as pd
import os

# --- SIMPLIFIED CONFIGURATION ---
# 'fks' and 'JUNCTION_SCHEMA' values are now just lists of entity names.
ENTITY_SCHEMA = {
    "track":         {"pk": "track", "prefix": "t", "fks": ["artist_credit", "recording", "medium"]},
    "recording":     {"pk": "recording", "prefix": "r", "fks": ["artist_credit"]},
    "medium":     {"pk": "medium", "prefix": "m", "fks": ["release"]},
    "release":     {"pk": "release", "prefix": "rl", "fks": ["release_group", "artist_credit"]},
    "release_group":     {"pk": "release_group", "prefix": "rg", "fks": ["artist_credit"]},
    "artist_credit": {"pk": "artist_credit", "prefix": "ac", "fks": []},
    "artist":        {"pk": "artist", "prefix": "a", "fks": ["area"]},
    "area":          {"pk": "area", "prefix": "b", "fks": []},
    "place":          {"pk": "place", "prefix": "p", "fks": ["area"]},
    "label": {"pk": "label", "prefix": "l", "fks": ["area"]}
}

JUNCTION_SCHEMA = {
    "artist_credit_name": ["artist_credit", "artist"]
}

INPUT_DIR = "./data/raw/music/50/"
OUTPUT_DIR = "./data/interim/music/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SchemaTransformer:
    def __init__(self, entities, junctions, input_dir, output_dir):
        self.entities = entities
        self.junctions = junctions
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.id_maps = {}

    def generate_mappings(self):
        print("--- Phase 1: Generating ID Mappings ---")
        for table, info in self.entities.items():
            df = pd.read_csv(os.path.join(self.input_dir, f"{table}.csv"))
            # Generate new IDs using the prefix and row index
            unique_ids = df[info['pk']].astype(str).unique()
            self.id_maps[table] = {
                old: f"{info['prefix']}{i}" for i, old in enumerate(unique_ids)}

    def transform_entities(self):
        print("\n--- Phase 2: Transforming Entities & Creating New Junctions ---")
        for table, info in self.entities.items():
            print(f"Processing entity: {table}")
            df = pd.read_csv(os.path.join(self.input_dir, f"{table}.csv"))
            pk_col = info['pk']

            # 1. Update Primary Key
            df[pk_col] = df[pk_col].astype(str).map(self.id_maps[table])

            # 2. Extract new junctions from attributes
            for entity_name in info['fks']:
                # The assumption: column name == table name
                junction_df = df[[pk_col, entity_name]].copy()

                # Map the FK column using the corresponding entity's map
                if entity_name in self.id_maps:
                    junction_df[entity_name] = junction_df[entity_name].astype(
                        str).map(self.id_maps[entity_name])

                # Save as a junction table and drop from main entity
                junction_df.dropna().to_csv(
                    os.path.join(self.output_dir, f"{table}_{entity_name}.csv"), index=False
                )
                df = df.drop(columns=[entity_name])

            # 3. Save Clean Entity Table
            df.to_csv(os.path.join(self.output_dir,
                      f"{table}.csv"), index=False)

            # 4. Handle matching/dups file automatically if present
            dup_path = os.path.join(self.input_dir, f"{table}_dups.csv")
            if os.path.exists(dup_path):
                self._map_file(dup_path, os.path.join(
                    self.output_dir, f"{table}_dups.csv"), [('1', table), ('2', table)])

    def transform_existing_junctions(self):
        print("\n--- Phase 3: Updating Existing Junction Tables ---")
        for table_name, columns in self.junctions.items():
            in_path = os.path.join(self.input_dir, f"{table_name}.csv")
            if os.path.exists(in_path):
                print(f"Updating junction: {table_name}")
                # For pre-existing junctions, we map the columns back to the entities of the same name
                mappings = [(col, col) for col in columns]
                self._map_file(in_path, os.path.join(
                    self.output_dir, f"{table_name}.csv"), mappings)

    def _map_file(self, in_path, out_path, column_target_pairs):
        """Generic helper to rewrite IDs in specific columns."""
        df = pd.read_csv(in_path)
        for col_name, target_entity in column_target_pairs:
            if col_name in df.columns and target_entity in self.id_maps:
                df[col_name] = df[col_name].astype(
                    str).map(self.id_maps[target_entity])
        df.dropna().to_csv(out_path, index=False)


# Run
transformer = SchemaTransformer(
    ENTITY_SCHEMA, JUNCTION_SCHEMA, INPUT_DIR, OUTPUT_DIR)
transformer.generate_mappings()
transformer.transform_entities()
transformer.transform_existing_junctions()
