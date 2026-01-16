from csv import DictReader
import csv
import uuid
import pandas as pd

from deepcem.utils import get_attrs_for_keys

dataset = "dirty/dblp-scholar"
task = f"{dataset}_lf"
lm = "ditto"
dataset_dir = f"./data/raw/deepmatcher/{dataset}"
output_norm_dir = f"./data/processed/normalized_dblp-scholar"
index_column = "id" 
delimiter = ","

train_fp = f"{dataset_dir}/train.csv" 
train_df = pd.read_csv(train_fp)

df_A = pd.read_csv(f"{dataset_dir}/tableA.csv")  # left
df_B = pd.read_csv(f"{dataset_dir}/tableB.csv")  # right


# MAIN TABLE: PUBLICATIONS
publications: dict[str, dict] = {}

# RELATION TABLE: AUTHORS (MENTIONS)
authors: dict[str, dict] = {}

# CONNCETION: AUTHOR -> PUBLICATIONS, PUBLICATION -> AUTHORS
pub_to_authors: dict[str, list] = {}
author_to_pubs: dict[str, list] = {}

with open(f"{dataset_dir}/tableA.csv", 'r') as f:
    
    dict_reader = DictReader(f)
    
    list_of_dict = list(dict_reader)

    for d in list_of_dict:
        if d["id"] not in publications:
            pub_key = d["id"]
        else: 
            pub_key = str(uuid.uuid4())

        publications[pub_key] = {
        "title": d["title"],
        "venue": d["venue"],
        "year": d["year"]
        }

        authors_list = d["authors"].split(delimiter)
        for a in authors_list:
            if a != "":
                a_key = str(uuid.uuid4())
                authors[a_key] = a.strip()

                if pub_key not in pub_to_authors:
                    pub_to_authors[pub_key] = [a_key]
                else: 
                    pub_to_authors[pub_key].append(a_key)

                if a_key not in author_to_pubs:
                    author_to_pubs[a_key] = [pub_key]
                else: 
                    author_to_pubs[a_key].append(pub_key)

file_path = f"{output_norm_dir}/publications.csv"

with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
    # Define the column headers
    fieldnames = ['id', 'title', 'venue', 'year']
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Loop through your dictionary and write rows
    for pub_id, data in publications.items():
        # Combine the ID with the rest of the data
        row = {'id': pub_id, 'title': data['title'], 'venue': data['venue'], 'year': data['year']}
        writer.writerow(row)

print(f"Data successfully written to {file_path}")

file_path = f"{output_norm_dir}/authors.csv"

with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
    # Define the column headers
    fieldnames = ['a_id', 'author']
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Loop through your dictionary and write rows
    for a_id, data in authors.items():
        # Combine the ID with the rest of the data
        row = {'a_id': a_id, 'author': data}
        writer.writerow(row)

print(f"Data successfully written to {file_path}")

file_path = f"{output_norm_dir}/pub_to_authors.csv"

with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
    # Define the column headers
    fieldnames = ['pub_id', 'author_list']
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Loop through your dictionary and write rows
    for pub_id, data in pub_to_authors.items():
        # Combine the ID with the rest of the data
        row = {'pub_id': pub_id, 'author_list': data}
        writer.writerow(row)

print(f"Data successfully written to {file_path}")

file_path = f"{output_norm_dir}/author_to_pubs.csv"

with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
    # Define the column headers
    fieldnames = ['a_id', 'pub_list']
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Loop through your dictionary and write rows
    for a_id, data in author_to_pubs.items():
        # Combine the ID with the rest of the data
        row = {'a_id': a_id, 'pub_list': data}
        writer.writerow(row)

print(f"Data successfully written to {file_path}")

# Convert to string once and index by 'id'
df_A_idx = df_A.set_index(index_column)
df_B_idx = df_B.set_index(index_column)

train = get_attrs_for_keys(df_A_idx, df_B_idx, train_df)

