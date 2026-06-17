import os
import shutil
import argparse
from pathlib import Path


def copy_job_directories(source_dir, dest_dir, job_id):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Create the destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    copied_count = 0

    # Format the prefix with an underscore to ensure strict matching
    # e.g., "9764501_" prevents matching "97645010_1"
    job_prefix = f"{job_id}_"

    print(
        f"Searching for array job ID: '{job_id}' (matches '{job_prefix}*' or exactly '{job_id}')...\n")

    # os.walk traverses the directory tree top-down
    for root, dirs, files in os.walk(source_path):
        for dir_name in dirs:
            # Check if the directory represents a task belonging to the array job ID
            if dir_name == job_id or dir_name.startswith(job_prefix):
                matched_dir_path = Path(root) / dir_name

                # Calculate relative path to maintain the nested folder structure
                # e.g., imdb/high/movie/finetune/9764501_2
                rel_path = matched_dir_path.relative_to(source_path)
                target_dir_path = dest_path / rel_path

                print(f"Found: {matched_dir_path}")

                try:
                    # dirs_exist_ok=True allows merging if the target structure already exists
                    shutil.copytree(matched_dir_path,
                                    target_dir_path, dirs_exist_ok=True)
                    print(f"  └── Copied to: {target_dir_path}")
                    copied_count += 1
                except Exception as e:
                    print(f"  └── Error copying {matched_dir_path}: {e}")

    print(f"\nDone. Successfully copied {copied_count} directories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively copy directories matching an array job ID.")
    parser.add_argument(
        "source", help="Path to the source directory (e.g., ~/deepcem/ditto_out)")
    parser.add_argument(
        "destination", help="Path to the destination directory (e.g., ~/deepcem/ditto_filtered)")
    parser.add_argument(
        "job_id", help="The array job ID to match (e.g., 9764501)")

    args = parser.parse_args()

    copy_job_directories(args.source, args.destination, args.job_id)
