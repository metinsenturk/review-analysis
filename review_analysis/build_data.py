import logging

from .data import folder_paths
from .data.data_scrapper import yelp_branches, scrappers

logger = logging.getLogger()

def get_competitor_reviews(start_index, end_index, download_competitors=False):
    sc = scrappers()

    if download_competitors:
        sc.yp_get_competitors(yelp_branches)
    
    try:
        sc.yp_get_competitor_reviews(start_index=start_index, end_index=end_index)
    except Exception as ex:
        logger.warning("error: " + ex)
    print("helo")

def view_current_businesses():
    message = ' Current yelp branches that is \n' \
              ' followed is listed in below.\n\n' + "\n".join(yelp_branches)
    
    logger.info(message)
    
    return print(message)