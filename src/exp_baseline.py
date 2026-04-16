import argparse
from pathlib import Path
import time

from baseline_config import REGISTRY, BaselineConfig
from experiment_config import DITTO_CONFIG
from ditto_wrapper import evaluate, finetune, refinetune
from logging_setup import ExperimentLogger, get_experiment_metadata, setup_logger


def main():
    # --- parse arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pollution", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_suffix", type=str, default="")
    args = parser.parse_args()
    dataset = args.dataset

    run_id, _ = get_experiment_metadata()
    # --- logging ---
    log = setup_logger(
        f"{run_id}_baseline_{dataset}_{args.pollution}_{args.seed}_{args.train_suffix}")
    log.info(f"BASELINE EXPERIMENT")
    log.info(
        f"Start Finetuning: Dataset: {args.dataset} {args.pollution}_ {args.seed}_ {args.train_suffix}")

    sql_log = ExperimentLogger("cem_results.db")

    # --- load config ---
    raw_config = REGISTRY[dataset]
    config: dict[str, BaselineConfig] = {
        name: entity.resolve_paths(
            args.dataset, args.pollution)
        for name, entity in raw_config.items()
    }

    for entity in config.values():
        start_time = time.perf_counter()

        # --- Finetune baseline model ---
        finetune(entity.model,
                 entity.dir_path, log, None, args.train_suffix, args.seed)

        input_path = f"{entity.dir_path}/test.txt"

        out_path = Path(
            f"./ditto_out/{dataset}/{args.pollution}/{entity.name}/{run_id}")
        Path(out_path).mkdir(parents=True, exist_ok=True)
        output_fp = f"{out_path}/baseline.jsonl"
        metrics_tuple = evaluate(entity.model, input_path,
                                 output_fp, log, input_path, entity.dir_path, [])

        metrics = {"accuracy": metrics_tuple[0], "precision": metrics_tuple[1],
                   "recall": metrics_tuple[2], "f1_score": metrics_tuple[3]}
        end_time = time.perf_counter()
        runtime = end_time - start_time

        # train_size
        try:
            if args.train_suffix == "":
                train_size = 1.0
            else:
                # Converts "_125" -> 125 -> 0.125 (assuming base is 1000)
                # Or adjust logic based on your specific naming convention
                train_size = round(
                    float(args.train_suffix.strip("_")) / 1000.0, 3)
        except ValueError:
            train_size = 1.0

        run_id = sql_log.log_run(
            run_id=run_id,
            dataset=dataset,
            entity=entity.name,
            train_size=train_size,
            model_type="baseline",
            batch_size=DITTO_CONFIG['batch_size'],
            max_len=DITTO_CONFIG['max_len'],
            learning_rate=DITTO_CONFIG['learning_rate'],
            epochs=DITTO_CONFIG['epochs'],
            lm=DITTO_CONFIG['lm'],
            neg_ratio=0,  # TODO
            seed=args.seed)

        # --- Log metrics ---
        sql_log.log_metrics(
            run_id=run_id,
            entity=entity.name,
            pollution=args.pollution,
            iteration=0,
            is_final=True,
            testset="test",
            metrics_dict=metrics,
            num_pairs=0,  # TODO Read from experiment config
            runtime=runtime)

        # --- Phase 2: Re-finetune, still only attributes ---
        # task_2 = f"{entity.model}_rel"
        # log.info(" ")
        # log.info("--- Start Finetune Phase 2: ---")
        # refinetune(task_2,
        #            entity.dir_path, entity.model, log, None, args.train_suffix, args.seed)

        # output_fp = f"{out_path}/baseline2.jsonl"

        # metrics_tuple = evaluate(task_2, input_path,
        #                          output_fp, log, input_path, entity.dir_path, [])
        # end_time = time.perf_counter()
        # runtime_phase2 = end_time - start_time

        # sql_log.log_metrics(
        #     run_id=run_id,
        #     pollution=args.pollution,
        #     iteration=1,
        #     is_final=True,
        #     testset="test",
        #     metrics_dict=metrics,
        #     num_pairs=0,  # TODO Read from experiment config
        #     runtime=runtime)


if __name__ == "__main__":
    main()
