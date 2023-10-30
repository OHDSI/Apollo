import dataclasses
import logging
import sys
from dataclasses import fields


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


def create_logger(log_file_name: str, clear_log_file: bool = False):
    """
    Sets up the root logger where it writes all logging events to file, and writing events at or above 'info' to
    console. Events are appended to the log file.

    Args:
        log_file_name: The name of the file where the log will be written to.
        clear_log_file: If true, the log file will be cleared before writing to it.
    """
    if clear_log_file:
        open(log_file_name, "w").close()
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.DEBUG)
    if not len(logger.handlers):
        _add_file_handler(logger=logger, log_file_name=log_file_name)
        _add_stream_handler(logger=logger)

    sys.excepthook = handle_exception


class _ConfigLogger(object):

    def log_config(self, config):
        logging.info("Config:")
        config.write(self)

    def write(self, data):
        line = data.strip()
        logging.info(line)


def log_config(config):
    config_logger = _ConfigLogger()
    config_logger.log_config(config)


def handle_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        logger = logging.getLogger()
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def log_settings(settings: dataclasses):
    logging.info("Settings:")
    for field in fields(settings):
        field_name = field.name
        field_value = getattr(settings, field_name)
        logging.info(f"- {field_name}: {field_value}")
