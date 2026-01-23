from deepcem.data_structures import Reference
def serialize_ditto(ri: Reference, rj: Reference):
    """
    Convert two pandas Series (records) into Ditto input format.

    Args:
        record_a (pd.Series): A row from dataframe A.
        record_b (pd.Series): A row from dataframe B.

    Returns:
        str: A Ditto-formatted string.
    """
    def serialize_record(record):
        parts = []
        for col, val in record.items():
            if val is None:
                continue
            parts.append(f'{col} : "{val}"')
        return " , ".join(parts)

    left = serialize_record(ri.attrs)
    right = serialize_record(rj.attrs)

    return f"{left}\t{right}\t0"