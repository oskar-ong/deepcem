import json
import pandas as pd
from typing import List

def load_to_multiindex(file_path: str, columns_to_drop: List[str] = None) -> pd.DataFrame:
    """
    Loads JSONL data and converts it into a MultiIndex DataFrame (left, right, metadata).
    """
    raw_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(tuple(json.loads(line)))

    # Unzip components
    left_list, right_list, labels = zip(*raw_data)

    # Create individual DataFrames
    df_left = pd.DataFrame(left_list)
    df_right = pd.DataFrame(right_list)
    df_label = pd.DataFrame(labels, columns=['match'])

    # Build MultiIndex
    df = pd.concat([df_left, df_right, df_label], axis=1)
    columns = (
        [('left', col) for col in df_left.columns] +
        [('right', col) for col in df_right.columns] +
        [('metadata', 'match')]
    )
    df.columns = pd.MultiIndex.from_tuples(columns)

    # Drop unwanted columns across both sides
    if columns_to_drop:
        # errors='ignore' ensures it doesn't crash if a column is already missing
        df = df.drop(columns=columns_to_drop, level=1, errors='ignore')

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

def serialize_to_ditto(df: pd.DataFrame, output_path: str = None) -> List[str]:
    """
    Converts a MultiIndex DataFrame into Ditto serialization format.
    """
    def format_row(row):
        # Helper to format one side into COL VAL strings
        def fmt(side):
            return " ".join([f"COL {k} VAL {v}" for k, v in row[side].items() if pd.notna(v)])
        
        return f"{fmt('left')}\t{fmt('right')}\t{row['metadata', 'match']}"

    # Use apply for faster row-wise processing
    ditto_lines = df.apply(format_row, axis=1).tolist()

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(ditto_lines) + "\n")
            
    return ditto_lines

def process_and_save_ditto(file_path: str, columns_to_drop: List[str], output_path: str):
    """
    Main pipeline function.
    """
    # 1. Load and Clean
    df = load_to_multiindex(file_path, columns_to_drop)
    print(f"Loaded and cleaned {len(df)} pairs.")
    
    # 2. Serialize and Save
    serialize_to_ditto(df, output_path)
    print(f"Ditto file saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    drop_list = ['tconst', 'cluster_id', 'block_key']
    process_and_save_ditto("./data/processed/imdb/movie/train.jsonl", drop_list, "./data/processed/imdb/movie/ditto/train.txt")
    process_and_save_ditto("./data/processed/imdb/movie/valid.jsonl", drop_list, "./data/processed/imdb/movie/ditto/valid.txt")
    process_and_save_ditto("./data/processed/imdb/movie/test.jsonl", drop_list, "./data/processed/imdb/movie/ditto/test.txt")
    drop_list = ['nconst', 'cluster_id', 'block_key']
    process_and_save_ditto("./data/processed/imdb/name/train.jsonl", drop_list, "./data/processed/imdb/name/ditto/train.txt")
    process_and_save_ditto("./data/processed/imdb/name/valid.jsonl", drop_list, "./data/processed/imdb/name/ditto/valid.txt")
    process_and_save_ditto("./data/processed/imdb/name/test.jsonl", drop_list, "./data/processed/imdb/name/ditto/test.txt")