import logging
import sys

# Create a custom logger
logger = logging.getLogger("product-ms")

# Set the threshold of logger to INFO
logger.setLevel(logging.DEBUG)

# Create a StreamHandler (Output to Console)
handler = logging.StreamHandler(sys.stdout)

# Create a Formatter
# Format: Timestamp | Level | Filename:Line | Message
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Add formatter to handler
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)