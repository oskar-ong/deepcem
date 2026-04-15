import argparse
from pathlib import Path
import time

from experiment_config import DITTO_CONFIG, REGISTRY, ExperimentConfig
from ditto_wrapper import evaluate, finetune, refinetune
from logging_setup import ExperimentLogger, setup_logger


def main():
    # --- parse arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pollution", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_suffix", type=str, default="")
    parser.add_argument("--bin", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset

    # --- logging ---
    log = setup_logger("finetune")
    log.info(f"Start Finetuning: Dataset: {args.dataset}")

    sql_log = ExperimentLogger("cem_results.db")

    # --- load config ---
    raw_config = REGISTRY[dataset]
    config: dict[str, ExperimentConfig] = {
        name: entity.resolve_paths(
            args.dataset, args.pollution, args.bin)
        for name, entity in raw_config.items()
    }

    configs_path = f"./models/ditto/configs.json"

    for entity in config.values():
        start_time = time.perf_counter()
        # get special tokens
        special_tokens = []
        for r in entity.relations:
            special_tokens.append(r.score_col)

        if args.train_suffix == "":
            task_base = f"{entity.model_base}_{args.seed}"
        else:
            task_base = f"{entity.model_base}_{args.seed}_{args.train_suffix}"
        # --- Phase 1: Initial Finetune, only attribute values ---
        finetune(task_base,
                 entity.empty_scores_dir, log, special_tokens, args.train_suffix, args.seed)

        input_path = f"{entity.empty_scores_dir}/test.txt"

        out_path = Path(
            f"./ditto_out/{dataset}/{args.pollution}/{entity.name}/finetune")
        Path(out_path).mkdir(parents=True, exist_ok=True)
        output_fp = f"{out_path}/phase1_eval.jsonl"
        evaluate(task_base, input_path, output_fp, log, input_path,
                 entity.empty_scores_dir, special_tokens)

        run_id = sql_log.log_run(
            dataset=dataset,
            entity=entity.name,
            train_size=args.train_suffix,
            model_type="1",
            batch_size=DITTO_CONFIG['batch_size'],
            max_len=DITTO_CONFIG['max_len'],
            learning_rate=DITTO_CONFIG['learning_rate'],
            epochs=DITTO_CONFIG['epochs'],
            lm=DITTO_CONFIG['lm'],
            neg_ratio=0,  # TODO
            seed=args.seed)

        end_time = time.perf_counter()
        runtime_phase1 = end_time - start_time

        task_rel = f"{task_base}_rel"
        # --- Phase 2: Re-finetune, include relaitonal scores ---
        refinetune(task_rel,
                   entity.injected_scores_dir, task_base, log, special_tokens, args.train_suffix, args.seed)

        input_path = f"{entity.injected_scores_dir}/test.txt"

        output_fp = f"{out_path}/phase2_eval.jsonl"
        evaluate(task_rel, input_path, output_fp, log, input_path,
                 entity.injected_scores_dir, special_tokens)
        end_time = time.perf_counter()
        runtime_phase2 = end_time - start_time


if __name__ == "__main__":
    main()
