import argparse
import json
from collections import Counter
import re


def process_file(filepath):
    """Reads a jsonl file and returns counts and a mapping of pairs to labels."""
    stats = Counter()
    pair_labels = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            data = json.loads(line)

            # Extract match label
            match_val = data.get('match')
            stats[match_val] += 1

            if "baseline.jsonl" in filepath:
                id_re_pattern = r"COL id VAL (\S+)"

                left_id_match = re.search(id_re_pattern, data["left"])
                right_id_match = re.search(id_re_pattern, data["right"])

                if left_id_match and right_id_match:
                    left_id = left_id_match.group(1)
                    right_id = right_id_match.group(1)
                else:
                    raise KeyError("Value for ID missing")

            else:
                left_id = data['left'].get('id')
                right_id = data['right'].get('id')
            pair_key = (left_id, right_id)

            pair_labels[pair_key] = {
                'match': match_val,
                'line_num': line_num,
                'data': data
            }

    return stats, pair_labels


def compare_jsonl_files(file1_path, file2_path):
    # Process both files
    stats1, labels1 = process_file(file1_path)
    stats2, labels2 = process_file(file2_path)

    # 1 & 2. Print Match Counts
    print(f"--- Statistics for {file1_path} ---")
    print(f"Match (1): {stats1[1]}")
    print(f"Non-Match (0): {stats1[0]}")

    print(f"\n--- Statistics for {file2_path} ---")
    print(f"Match (1): {stats2[1]}")
    print(f"Non-Match (0): {stats2[0]}")

    # 3. Find different match labels
    print("\n--- Discrepancies (Different Match Labels) ---")
    diff_count = 0

    # We iterate through keys present in both files
    common_keys = set(labels1.keys()) & set(labels2.keys())

    for key in common_keys:
        label1 = labels1[key]['match']
        label2 = labels2[key]['match']

        if label1 != label2:
            diff_count += 1
            left_id, right_id = key
            print(f"Pair: {left_id} <-> {right_id}")
            print(
                f"  File 1 (Line {labels1[key]['line_num']}): match={label1}")
            print(
                f"  File 2 (Line {labels2[key]['line_num']}): match={label2}")
            print("-" * 30)

    if diff_count == 0:
        print("No label discrepancies found between common pairs.")
    else:
        print(f"Total differences found: {diff_count}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("fp1", type=str)
    # parser.add_argument("fp2", type=str)
    # args = parser.parse_args()
    # fp1 = args.fp1
    # fp2 = args.fp2

    # job : 9790002_15
    # imdb, high
    # fp1 = "ditto_out/baseline.jsonl"
    # fp2 = "ditto_out/movie_3_conv_results.jsonl"

    # job : 9798848_15
    # music, high

    fp1 = "music_results/baseline.jsonl"
    fp2 = "music_results/track_3_conv_results.jsonl"

    compare_jsonl_files(fp1, fp2)
