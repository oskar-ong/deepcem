import argparse
from pathlib import Path
import time

from baseline_config import REGISTRY, BaselineConfig
from experiment_config import DITTO_CONFIG
from ditto_wrapper import evaluate, finetune
from logging_setup import ExperimentLogger, setup_logger


def main():
    # --- parse arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pollution", type=str)
    args = parser.parse_args()
    dataset = args.dataset

    # --- logging ---
    log = setup_logger("baseline_finetune")
    log.info(f"BASELINE EXPERIMENT")
    log.info(f"Start Finetuning: Dataset: {args.dataset}")

    sql_log = ExperimentLogger("cem_results.db")

    # --- load config ---
    raw_config = REGISTRY[dataset]
    config: dict[str, BaselineConfig] = {
        name: entity.resolve_paths(
            args.dataset, args.pollution)
        for name, entity in raw_config.items()
    }

    configs_path = f"./models/ditto/configs.json"

    for entity in config.values():
        start_time = time.perf_counter()

        # --- Finetune baseline model ---
        finetune(configs_path, entity.model,
                 entity.dir_path, log, None)

        input_path = f"{entity.dir_path}/test.txt"

        out_path = Path(
            f"./ditto_out/{dataset}/{args.pollution}/{entity.name}")
        Path(out_path).mkdir(parents=True, exist_ok=True)
        output_fp = f"{out_path}/baseline.jsonl"
        metrics = evaluate(entity.model, input_path,
                           output_fp, log, input_path)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        sql_log.log_run(
            dataset=dataset,
            model_type="baseline",
            batch_size=DITTO_CONFIG['batch_size'],
            max_len=DITTO_CONFIG['max_len'],
            learning_rate=DITTO_CONFIG['learning_rate'],
            epochs=DITTO_CONFIG['epochs'],
            lm=DITTO_CONFIG['lm'],
            neg_ratio=0,  # TODO
            seed=0)  # TODO

        # --- Log metrics ---
        sql_log.log_metrics(
            pollution=args.pollution,
            iteration=0,
            entity=entity.name,
            testset_type="baseline",
            metrics_dict=metrics,
            num_pairs=0,  # TODO Read from experiment config
            runtime=runtime)


if __name__ == "__main__":
    main()
