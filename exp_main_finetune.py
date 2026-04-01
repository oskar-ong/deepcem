import argparse
from pathlib import Path

from experiment_config import REGISTRY, ExperimentConfig
from src.ditto_wrapper import evaluate, finetune, refinetune
from src.logging_setup import ExperimentLogger, setup_logger


def main():
    # --- parse arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pollution", type=str)
    args = parser.parse_args()
    dataset = args.dataset

    # --- logging ---
    log = setup_logger("finetune")
    sql_log = ExperimentLogger("entity_resolution_results.db")
    log = setup_logger("exp_main_exp-matching")
    log.info(f"Start Finetuning: Dataset: {args.dataset}")
    run_params = {'model': 'Ditto', 'lr': 3e-5,
                  'batch_size': 16}  # Read from experiment config?
    run_id = sql_log.log_run(run_params)

    # --- load config ---
    raw_config = REGISTRY[dataset]
    config: dict[str, ExperimentConfig] = {
        name: entity.resolve_paths(
            args.dataset, args.pollution)
        for name, entity in raw_config.items()
    }

    configs_path = f"./models/ditto/configs.json"

    for entity in config.values():
        # --- Phase 1: Initial Finetune, only attribute values ---
        finetune(configs_path, entity.model_base, entity.empty_scores_dir, log)

        input_path = f"{entity.empty_scores_dir}/test.txt"

        out_path = Path(
            f"./ditto_out/{dataset}/{args.pollution}/{entity.name}")
        Path(out_path).mkdir(parents=True, exist_ok=True)
        output_fp = f"{out_path}/phase1_eval.jsonl"
        evaluate(entity.model_base, input_path, output_fp, log, input_path)

        # --- Phase 2: Re-finetune, include relaitonal scores ---
        refinetune(configs_path, entity.model,
                   entity.injected_scores_dir, entity.model_base, log)

        input_path = f"{entity.injected_scores_dir}/test.txt"

        out_path = Path(
            f"./ditto_out/{dataset}/{args.pollution}/{entity.name}")
        Path(out_path).mkdir(parents=True, exist_ok=True)
        output_fp = f"{out_path}/phase2_eval.jsonl"
        evaluate(entity.model, input_path, output_fp, log, input_path)


if __name__ == "__main__":
    main()
