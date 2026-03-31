from datetime import datetime
import logging
import os
import sqlite3


def setup_logger(base_name="CollectiveER"):
    # Generate timestamp: e.g., 20260306_1012
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    slurm_job_id TEXT,
                    model_type TEXT,
                    learning_rate REAL,
                    batch_size INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id INTEGER,
                    epoch INTEGER,
                    entity TEXT,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
            """)

    def log_run(self, params):
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.cursor()
            query = """INSERT INTO runs (timestamp, slurm_job_id, model_type, learning_rate, batch_size) 
                       VALUES (?, ?, ?, ?, ?)"""
            cursor.execute(query, (
                datetime.now().isoformat(),
                os.getenv("SLURM_JOB_ID", "local"),
                params['model'],
                params['lr'],
                params['batch_size']
            ))
            return cursor.lastrowid

    def log_metrics(self, run_id, epoch, entity, metrics_dict):
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            query = "INSERT INTO metrics (run_id, epoch, entity, precision, recall, f1_score) VALUES (?, ?, ?, ?, ?, ?)"
            conn.execute(query, (
                run_id,
                epoch,
                entity,
                metrics_dict['precision'],
                metrics_dict['recall'],
                metrics_dict['f1_score']
            ))
