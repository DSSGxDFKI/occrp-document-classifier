import datetime
import logging
import os

import tensorflow as tf

from occrplib.config import settings

os.makedirs(settings.DATA_PATH_LOGS, exist_ok=True)

LOGGING_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}
LOGGING_FORMAT = r"%(asctime)s [%(filename)s:%(lineno)s -%(funcName)20s()] - %(name)s - %(levelname)s - %(message)s"
LOGGING_FILENAME = datetime.datetime.now().strftime(os.path.join(settings.DATA_PATH_LOGS, r"%Y_%m_%d_%H_%M_%S.log"))

TENSORFLOW_LOGGING_LEVEL = logging.ERROR
tf.get_logger().setLevel(TENSORFLOW_LOGGING_LEVEL)
