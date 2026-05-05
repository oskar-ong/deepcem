import uuid

import pandas as pd
import os

ENTITY_SCHEMA_POKEMON = {
    "pokemon":         {"pk": "pokemon", "prefix": "p", "fks": ["species"], "denormalize": False},
    "ability":     {"pk": "ability", "prefix": "a", "fks": [], "denormalize": True},
    "species": {"pk": "species", "prefix": "s", "fks": [], "denormalize": True},
    "item": {"pk": "item", "prefix": "i", "fks": [], "denormalize": True},
    "move": {"pk": "move", "prefix": "m", "fks": [], "denormalize": True}
}

JUNCTION_SCHEMA_POKEMON = {
    "poke_ability": [
        ("pokemon", "pokemon"),
        ("ability", "ability")
    ],
    "poke_item": [
        ("pokemon", "pokemon"),
        ("item", "item")
    ],
    "poke_move": [
        ("pokemon", "pokemon"),
        ("move", "move")
    ]
}

INPUT_DIR_POKEMON = "./data/raw/pokemon/50/"
OUTPUT_DIR_POKEMON = "./data/interim/pokemon/"

os.makedirs(OUTPUT_DIR_POKEMON, exist_ok=True)


class SchemaTransformer:
    def __init__(self, entities, junctions, input_dir, output_dir):
        self.entities = entities
        self.junctions = junctions
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.id_maps = {}
        self.dfs = {}

    def denormalize(self):
        for table, info in self.entities.items():
            df = pd.read_csv(os.path.join(self.input_dir, f"{table}.csv"))

            if info["denormalize"] == True:

                df_desc = pd.read_csv(os.path.join(
                    self.input_dir, f"{table}_desc.csv"))
                df_name = pd.read_csv(os.path.join(
                    self.input_dir, f"{table}_name.csv"))

                df_desc_filtered = df_desc[df_desc["language"] == 9]
                df_name_filtered = df_name[df_name["local_language"] == 9]

                df = df.merge(
                    df_desc_filtered[[info["pk"], "flavor_text"]], on=info["pk"], how="left")
                df = df.merge(
                    df_name_filtered[[info["pk"], "name"]], on=info["pk"], how="left")

            self.dfs[table] = df

    def generate_mappings(self):
        print("--- Phase 1: Generating ID Mappings ---")
        for table, info in self.entities.items():
            # df = pd.read_csv(os.path.join(self.input_dir, f"{table}.csv"))
            df = self.dfs[table]
            # Generate new IDs using the prefix and row index
            unique_ids = df[info['pk']].astype(str).unique()
            self.id_maps[table] = {
                old: f"{info['prefix']}_{uuid.uuid4().hex[:8]}"
                for old in unique_ids
            }

    def transform_entities(self):
        print("\n--- Phase 2: Transforming Entities & Creating New Junctions ---")
        for table, info in self.entities.items():
            print(f"Processing entity: {table}")
            # df = pd.read_csv(os.path.join(self.input_dir, f"{table}.csv"))
            df = self.dfs[table]
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
        for table_name, mappings in self.junctions.items():
            in_path = os.path.join(self.input_dir, f"{table_name}.csv")
            if os.path.exists(in_path):
                print(f"Updating junction: {table_name}")
                # 'mappings' is now expected to be a list of (column, entity) tuples
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
    ENTITY_SCHEMA_POKEMON, JUNCTION_SCHEMA_POKEMON, INPUT_DIR_POKEMON, OUTPUT_DIR_POKEMON)
transformer.denormalize()
transformer.generate_mappings()
transformer.transform_entities()
transformer.transform_existing_junctions()
