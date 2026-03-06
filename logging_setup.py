import datetime
import logging


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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    return logger