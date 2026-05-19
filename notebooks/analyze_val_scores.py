import pathlib
import re

from results_analysis import run_sql_query


keywords = ["Dataset:", "JOB", ]


def parse_keywords(input_filename, output_filename):
    try:
        with open(input_filename, 'r', encoding='utf-8') as input_file, \
                open(output_filename, 'w', encoding='utf-8') as output_file:
            best_dev_f1 = 0.0
            task = "default"
            new_task = ""
            for line in input_file:

                if any(key in line for key in keywords):
                    output_file.write(line)

                if "best_dev_f1" in line:

                    dev_f1_match = re.search(r"best_dev_f1 ([\d.]+)", line)
                    if dev_f1_match:
                        best_dev_f1 = float(dev_f1_match.group(1))

                if "hp.task:" in line:

                    task_match = re.search(r"hp.task: ([\w]+)", line)
                    if task_match:
                        new_task = str(task_match.group(1))
                    if task != new_task:
                        output_file.write(f"{new_task}\n")
                        task = new_task

                if "epoch 10:" in line:
                    output_file.write(f"{best_dev_f1} \n")

                    task = parse_task(task)
                    return {task: best_dev_f1}
                    # print(f"best best_dev_f1: {best_dev_f1}")

                    # TODO: Save best dev f1 for current task
                    # Add best_dev_f1s to stack
                    # if key contains "model"
                    # pop stack -> last best dev f1

                if "EXPERIMENT" in line:
                    break

        print(f"Parsing complete. Matching lines saved to {output_filename}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")


def parse_filename(filename: str):
    split_by_dash = filename.split("-")
    split_by_dot = split_by_dash[1].split(".")[0]
    result = f"{split_by_dash[1]}_{split_by_dot[0]}"
    return result


def process_directory(directory_path, jobid):
    # Convert string path to a Path object
    base_dir = pathlib.Path(directory_path)

    output_dir = base_dir / "parsed_results"
    output_dir.mkdir(exist_ok=True)

    results = []

    for file_path in base_dir.iterdir():

        if file_path.is_file() and file_path.name.startswith(f"e2e-{jobid}"):
            print(f"Processing: {file_path.name}...")

            output_file = output_dir / f"parsed_{file_path.name}"

            # parse keywords returns a dict[str, float]
            val_scores = parse_keywords(file_path, output_file)

            array_job = parse_filename(file_path.name)
            print(array_job)
            df = query_exp(array_job)
            print(df.shape)

            df_pivoted = df.pivot(
                index="entity", columns="metric_type", values="f1_score"
            ).reset_index()

            for col in ["baseline", "test"]:
                if col not in df_pivoted.columns:
                    df_pivoted[col] = None

            df_pivoted["val_score"] = df_pivoted["entity"].map(val_scores)

            file_results = list(
                df_pivoted[
                    ["entity", "val_score", "baseline", "test"]
                ].itertuples(index=False, name=None)
            )

            results.extend(file_results)
    return results


def query_exp(job_id):
    query = f"""SELECT * FROM metrics WHERE 
    is_final = 1 AND run_id = '{job_id}';"""

    df = run_sql_query(query)
    # df.drop(columns=["batch_size", "max_len",
    #         "learning_rate", "lm", "neg_ratio"], inplace=True)
    return df


if __name__ == "__main__":
    target_directory = "./logs"
    process_directory(target_directory, prefix="jobid")
