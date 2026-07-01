import os
from importlib.util import find_spec

from configs.config import LOG_LEVEL


if find_spec('loguru'):
    os.environ['LOGURU_LEVEL'] = LOG_LEVEL
    from loguru import logger
else:
    import logging
    format = '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s'
    logging.basicConfig(
        level=LOG_LEVEL,
        datefmt='%Y-%m-%d %H:%M:%S',
        format=format,
    )
    logger = logging.getLogger()
