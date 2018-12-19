import sys
sys.path.append('../')

import utilities
from data.data_scrapper import scrappers, yelp_branches
from data.data_builder import create_dataset

if __name__ == '__main__':
    create_dataset('././data/raw/yp_kimos-maui-lahaina_rws.json', '././data/processed/dataset1.csv')