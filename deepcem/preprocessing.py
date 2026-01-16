from csv import DictReader
import csv
import uuid

import pandas as pd

from deepcem.serialize import sanitize_value

def normalize(fp, pk="id", m2m="authors", delimiter=','):
    # MAIN TABLE: PUBLICATIONS
    publications: dict[str, dict] = {}

    # RELATION TABLE: AUTHORS (MENTIONS)
    authors: dict[str, dict] = {}

    # CONNECTION: AUTHOR -> PUBLICATIONS, PUBLICATION -> AUTHORS
    pub_to_authors: dict[str, list] = {}
    author_to_pubs: dict[str, list] = {}

    with open(fp, 'r', encoding='utf-8') as f:
        
        dict_reader = DictReader(f)
        
        list_of_dict = list(dict_reader)

        for d in list_of_dict:
            if d[pk] not in publications:
                pub_key = d[pk]
            else: 
                pub_key = str(uuid.uuid4())

            # assign key-value for each pair in current dict, except primary key and many2many
            publications[pub_key] = {
                k: v for k, v in d.items() if k not in [pk, m2m]
            }    

            # if many2many is assigned as function parameter and m2m value is assigned to current dict
            if m2m in d and d[m2m]:
                # split authors string and assign each stripped name to new value and append to list
                names = [n.strip() for n in d[m2m].split(delimiter) if n.strip()]
                for n in names:
                    a_key = str(uuid.uuid4())
                    authors[a_key] = {"name": n}

                    pub_to_authors.setdefault(pub_key, []).append(a_key)
                    author_to_pubs.setdefault(a_key, []).append(pub_key)
    return publications, authors, pub_to_authors, author_to_pubs

def reduce(pairs, src, out, id_src, id_pairs):
    # 1. Collect all unique IDs from both columns in pairs.csv
    selected_ids = set()
    with open(pairs, 'r', encoding='utf-8') as f:
        reader = DictReader(f)
        for row in reader:
            selected_ids.add(row[id_pairs])

    # 2. Read the full publications file and write only the matches

    with open(src, 'r', encoding='utf-8') as f_in, \
         open(out, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = DictReader(f_in)
        # Automatically use the headers from the source file
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        
        writer.writeheader()
        
        for row in reader:
            if row[id_src] in selected_ids:
                writer.writerow(row)

def build_ditto_file_from_labels(pairs_fp, table_a, table_b, output_dir, split):

    pairs = pd.read_csv(pairs_fp, dtype={'left': str, 'right': str})

    pairs['ltable_id'] = pairs['ltable_id'].astype(str).str.strip()
    pairs['rtable_id'] = pairs['rtable_id'].astype(str).str.strip()

    pairs['row_left'] = pairs['ltable_id'].map(table_a)
    pairs['row_right'] = pairs['rtable_id'].map(table_b)

    with open(f"{output_dir}/{split}.txt","w", encoding="utf-8") as f:
        
        for index, row in pairs.iterrows():
            data_left = row['row_left']
            data_right = row['row_right']
            label = row['label']
            #print(f"left: {data_left}, right: {data_right}, label: {label}")
            str_left = " ".join(f"COL {k} VAL {sanitize_value(v)}" for k, v in data_left.items())
            str_right = " ".join(f"COL {k} VAL {sanitize_value(v)}" for k, v in data_right.items())
            out_row = f"{str_left}\t{str_right}\t{label}\n"
            f.write(out_row)



# Usage
#filter_publications('pairs.csv', 'publications.csv', 'publications_reduced.csv')
