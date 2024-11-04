import logging


# create logger
def get_logger(name: str, loglevel: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)

    # create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)

    # create formatter
    formatter = logging.Formatter("%(name)s | %(levelname)s: %(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger
