# Logging configuration
import logging
import sys

# Create logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

# Create console handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add formatter to handler
stream_handler.setFormatter(formatter)

# Add handler to logger
if not logger.handlers:
    logger.addHandler(stream_handler)

# Prevent propagation to root logger
logger.propagate = False
