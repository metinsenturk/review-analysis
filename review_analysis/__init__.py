import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    handlers=[
        logging.FileHandler('../data/logs/logs_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("review_analysis")
logger.info(__name__)

from . import build_data
from . import model_data
from . import process_data
from . import app

__all__ = [app, build_data, model_data, process_data]
