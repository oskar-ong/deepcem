def resolve_train_size(train_suffix):
    try:
        if train_suffix == "":
            train_size = 1.0
        else:
            # Converts "_125" -> 125 -> 0.125 (assuming base is 1000)
            # Or adjust logic based on your specific naming convention
            train_size = round(
                float(train_suffix.strip("_")) / 1000.0, 3)
    except ValueError:
        train_size = 1.0
    return train_size
