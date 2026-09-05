import logging

def get_dist_info():
    rank = 0
    world_size = 1
    return rank, world_size


def root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(__name__.split(".")[0])
    # if the logger has been initialized, just return it
    # if logger.hasHandlers():
    #     return logger

    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel("ERROR")
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    
    return logger