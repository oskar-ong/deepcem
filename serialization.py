import pandas as pd

def serialize_to_ditto(df, output_path=None):
    """
    Serializes a MultiIndex DataFrame into the Ditto text format.
    Format: [COL attr VAL val ...] \t [COL attr VAL val ...] \t label
    """
    ditto_lines = []
    
    # We iterate over rows. df['left'] and df['right'] return sub-DataFrames
    # containing only the relevant attributes.
    for _, row in df.iterrows():
        # Helper to format a single side (left or right)
        def format_entry(side_series):
            parts = []
            for attr, val in side_series.items():
                # Ensure value is string and handle potential NaNs
                val_str = str(val) if pd.notna(val) else ""
                parts.append(f"COL {attr} VAL {val_str}")
            return " ".join(parts)

        # Process the three components
        entry_1 = format_entry(row['left'])
        entry_2 = format_entry(row['right'])
        label = str(row['metadata', 'match'])
        
        # Combine into the Ditto line format
        ditto_line = f"{entry_1}\t{entry_2}\t{label}"
        ditto_lines.append(ditto_line)
    
    # Optionally save to file
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(ditto_lines) + "\n")
            
    return ditto_lines

# Usage:
ditto_data = serialize_to_ditto(df, "train.txt")