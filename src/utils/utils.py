import logging

def setup_logger(name: str, log_file: str, level = logging.INFO):
    """
    Creates and configures logger

    Args:
        name (str): logger name
        log_file (str): File to log to
        level (_type_, optional): Logging level
    """
    
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levlename)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger