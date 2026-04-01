from datetime import datetime
import logging
import os
import socket
import sqlite3
from typing import Tuple


def setup_logger(base_name="CollectiveER"):
    # Generate timestamp: e.g., 20260306_1012
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_filename = f"logs/{base_name}_{timestamp}.log"

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
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if job_id:
        run_id = job_id
        if task_id:
            run_id += f"_{task_id}"
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
        self.run_id, self.env_type = get_experiment_metadata()

    def _setup_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # WAL mode allows multiple readers and one writer concurrently
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    dataset TEXT,
                    model_type TEXT,
                    batch_size INTEGER,
                    max_len INTEGER,
                    learning_rate REAL,
                    epochs INTEGER,
                    lm TEXT,
                    neg_ratio INTEGER, 
                    seed INTEGER
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT,
                    pollution TEXT,
                    iteration INTEGER,
                    entity TEXT,
                    testset_type TEXT,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    num_pairs INTEGER,
                    runtime REAL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
            """)

    def log_run(self, dataset, model_type, batch_size, max_len, learning_rate, epochs, lm, neg_ratio, seed):
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            query = """INSERT OR IGNORE INTO runs (run_id, timestamp, dataset, model_type, batch_size, max_len, learning_rate, epochs, lm, neg_ratio, seed) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            conn.execute(query, (self.run_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dataset, model_type, batch_size, max_len, learning_rate, epochs, lm, neg_ratio, seed
                                 ))
            return self.run_id

    def log_metrics(self, pollution, iteration, entity, testset_type, metrics_dict, num_pairs, runtime):
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            query = """INSERT INTO metrics (run_id, pollution, iteration, entity, testset_type, precision, recall, f1_score, num_pairs, runtime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            conn.execute(query, (
                self.run_id,
                pollution,
                iteration,
                entity,
                testset_type,
                metrics_dict['precision'],
                metrics_dict['recall'],
                metrics_dict['f1_score'],
                num_pairs,
                runtime
            ))
