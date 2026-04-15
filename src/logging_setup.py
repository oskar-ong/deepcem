from datetime import datetime
import logging
import os
import socket
import sqlite3
from typing import Tuple


def setup_logger(base_name="CollectiveER"):
    # Generate timestamp: e.g., 20260306_1012
    timestamp = datetime.now().strftime("%m-%d-%H:%M")
    log_filename = f"logs/{timestamp}_{base_name}.log"

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
        short_ts = datetime.now().strftime("%H%M%S")
        if task_id:
            run_id += f"_{task_id}_{short_ts}"
        else:
            run_id += f"_{short_ts}"
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
                    entity TEXT,
                    train_size REAL,
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
                    is_final BOOLEAN,
                    testset TEXT,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    num_pairs INTEGER,
                    runtime REAL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
            """)

            conn.execute("""
                            CREATE VIEW IF NOT EXISTS experiment_summary AS
                            SELECT 
                                r.dataset,
                                r.entity, 
                                r.train_size, 
                                r.seed, 
                                r.lm,
                                m.pollution, 
                                m.iteration, 
                                m.f1_score,
                                m.precision,
                                m.recall,
                                m.is_final,
                                r.run_id
                            FROM metrics m
                            JOIN runs r ON m.run_id = r.run_id
                        """)

    def log_run(self, dataset, entity, train_size, model_type, batch_size, max_len, learning_rate, epochs, lm, neg_ratio, seed):
        with sqlite3.connect(self.db_path, timeout=60) as conn:
            run_id, self.env_type = get_experiment_metadata()
            run_id = f"{run_id}_{entity}"
            query = """INSERT OR IGNORE INTO runs (run_id, timestamp, dataset, entity, train_size, model_type, batch_size, max_len, learning_rate, epochs, lm, neg_ratio, seed) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            conn.execute(query, (run_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dataset, entity, train_size, model_type, batch_size, max_len, learning_rate, epochs, lm, neg_ratio, seed
                                 ))
            return run_id

    def log_metrics(self, run_id, pollution, iteration, is_final, testset, metrics_dict, num_pairs, runtime):
        with sqlite3.connect(self.db_path, timeout=60) as conn:
            query = """INSERT INTO metrics (run_id, pollution, iteration, is_final, testset, precision, recall, f1_score, num_pairs, runtime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            conn.execute(query, (
                run_id,
                pollution,
                iteration,
                is_final,
                testset,
                metrics_dict['precision'],
                metrics_dict['recall'],
                metrics_dict['f1_score'],
                num_pairs,
                runtime
            ))
