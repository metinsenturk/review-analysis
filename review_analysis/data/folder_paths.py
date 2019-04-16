def yp_raw_reviews(yelp_username):
    """
    Reviews file of the given yelp_username.
    File Type: JSON
    """
    return '../data/raw/yp_{}_rws.json'.format(yelp_username)

def yp_raw_businesses(yelp_username):
    """
    Competitors file of the given yelp_username.
    File Type: CSV
    """
    return '../data/raw/{}_competitors.csv'.format(yelp_username)

def yp_processed_reviews(yelp_username):
    """
    Raw form of reviews that contains ony status and reviews.
    File Type: CSV
    """
    return '../data/processed/{}.csv'.format(yelp_username)

def yp_raw_competitors(data_path):
    """
    The file contains the list of business objects.
    File Type: JSON
    """
    return f'{data_path}/yp_competitors.json'

def yp_raw_competitors_reviews(data_path):
    """
    The file contains the reviews of all businesses that is supplied at the time of generation.
    File Type: CSV
    """
    return f'{data_path}/yp_competitors_rws.csv'