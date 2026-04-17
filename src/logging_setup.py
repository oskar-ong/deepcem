from datetime import datetime
import logging
import os
import socket
import sqlite3
from typing import Tuple


def setup_logger(base_name="CollectiveER"):
    # Generate timestamp: e.g., 20260306_1012
    log_filename = f"logs/{base_name}.log"
    # Create logger
    logger = logging.getLogger(base_name)
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_filename)

    # Create format for the logs
    # Format: Time - Name - Level - Message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


def get_experiment_metadata(override_id=None) -> Tuple[str, str]:
    if override_id:
        return override_id, "local"

    job_id = os.environ.get("SLURM_JOB_ID")
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if array_job_id and task_id:
        run_id = f"{array_job_id}_{task_id}"
        env = "hpc_array"
    elif job_id:
        run_id = job_id
        env = "hpc"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname()
        run_id = f"local_{hostname}_{timestamp}"
        env = "local"
    return run_id, env


class ExperimentLogger:
    def __init__(self, db_path="experiments.db"):
        self.db_path = db_path
        self._setup_db()

    def _setup_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # WAL mode allows multiple readers and one writer concurrently
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    dataset TEXT,
                    train_size REAL,
                    pollution TEXT,
                    seed INTEGER,
                    batch_size INTEGER,
                    max_len INTEGER,
                    learning_rate REAL,
                    epochs INTEGER,
                    lm TEXT,
                    neg_ratio INTEGER
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT,
                    entity TEXT,
                    metric_type TEXT,
                    iteration INTEGER,
                    is_final BOOLEAN,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    num_pairs INTEGER,
                    runtime REAL
                )
            """)

    def log_run(self, run_id, dataset, train_size, pollution, batch_size, max_len, learning_rate, epochs, lm, neg_ratio, seed):
        with sqlite3.connect(self.db_path, timeout=60) as conn:
            query = """INSERT OR IGNORE INTO runs (
            run_id, 
            timestamp, 
            dataset, 
            train_size, 
            pollution, 
            seed, 
            batch_size, 
            max_len, 
            learning_rate, 
            epochs, 
            lm, 
            neg_ratio) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            conn.execute(query, (run_id,
                                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 dataset,
                                 train_size,
                                 pollution,
                                 seed,
                                 batch_size,
                                 max_len,
                                 learning_rate,
                                 epochs,
                                 lm,
                                 neg_ratio,
                                 ))
            return run_id

    def log_metrics(self, run_id, entity, iteration, is_final, metric_type, metrics_dict, num_pairs, runtime):
        with sqlite3.connect(self.db_path, timeout=60) as conn:
            query = """INSERT INTO metrics (
            run_id, 
            entity, 
            metric_type, 
            iteration, 
            is_final, 
            precision, 
            recall, 
            f1_score, 
            num_pairs, 
            runtime) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

            conn.execute(query, (
                run_id,
                entity,
                metric_type,
                iteration,
                is_final,
                metrics_dict['precision'],
                metrics_dict['recall'],
                metrics_dict['f1_score'],
                num_pairs,
                runtime
            ))
