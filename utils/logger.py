import logging


def _add_stream_handler(logger: logging.Logger):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger


def _add_file_handler(logger: logging.Logger, log_file_name: str):
    file_handler = logging.FileHandler(log_file_name, mode="a")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def create_logger(log_file_name: str):
    """
    Sets up the root logger where it writes all logging events to file, and writing events at or above 'info' to
    console. Events are appended to the log file.

    Args:
        log_file_name: The name of the file where the log will be written to.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not len(logger.handlers):
        _add_file_handler(logger=logger, log_file_name=log_file_name)
        _add_stream_handler(logger=logger)
