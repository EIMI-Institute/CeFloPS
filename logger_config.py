import logging


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove all handlers associated with this logger to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add a console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    def info_like_print(*args, log_level=logging.INFO):
        if logger.isEnabledFor(log_level):
            formatted_message = " ".join(str(arg) for arg in args)
            logger.log(log_level, formatted_message)

    logger.print = info_like_print

    return logger


def set_global_log_level(new_level):
    # Set the level for the root logger
    logging.getLogger().setLevel(new_level)

    # Set the level for all existing loggers
    logger_dict = logging.root.manager.loggerDict
    for logger_name, logger in logger_dict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(new_level)
