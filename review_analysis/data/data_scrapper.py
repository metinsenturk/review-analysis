# system imports
import os
import sys
import json
import datetime
import random
import time
import csv
import logging

# 3rd party imports
import requests
import pandas as pd
import numpy as np
from yelpapi import YelpAPI
# from twitter import Twitter, OAuth
from searchtweets import load_credentials, ResultStream, gen_rule_payload
from bs4 import BeautifulSoup

## folder imports
from data import folder_paths as fp
from data.credentials import get_credidentials

topics = {
    "price",
    "ambience",
    "food",
    "bar",
    "view",
    "parking",
    "staff",
    "service",
    "music",
    "cleanliness",
    "time",
    "customer_service"
}

twitter_users = [
    'KimosRestaurant',
    'JakesInDelMar',
    'SunnysideResort',
    'dukeshb',
    'DukesLaJolla',
    'DukesMalibu',
    'DukesBeachHouse',
    'DukesInKauai',
    'DukesWaikiki',
    'hulagrillwaiks',
    'HulaGrillMaui',
    'KeokisParadise',
    'LeilanisMaui'
]

data_path = ".././data/raw"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr = logging.FileHandler(f"{data_path}/logs_scrapper.log")
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

yelp_branches = [
    'kimos-maui-lahaina',
    'leilanis-lahaina-2',
    'hula-grill-kaanapali-lahaina-2',
    'sunnyside-tahoe-city-2',
    'dukes-huntington-beach-huntington-beach-2',
    'dukes-la-jolla-la-jolla',
    'dukes-malibu-malibu-2',
    'dukes-beach-house-lahaina',
    'dukes-kauai-lihue-3',
    'dukes-waikiki-honolulu-2',
    'hula-grill-waikiki-honolulu-3',
    'keokis-paradise-koloa',
]

class scrappers:
    
    def __init__(self):
        __dir_path = os.path.dirname(os.path.realpath(__file__))
        credentials = get_credidentials()
        self.twitter_premium_api = load_credentials(
            filename="{}/{}".format(__dir_path,"twitter_keys.yaml"),
            yaml_key="search_tweets_api_30day")
        # self.twitter_api = Twitter(auth=OAuth(
        #     consumer_key=credentials['twitter']['consumer_key'],
        #     consumer_secret=credentials['twitter']['consumer_secret'],
        #     token=credentials['twitter']['access_token_key'],
        #     token_secret=credentials['twitter']['access_token_secret']
        # ))
        self.yelp_api = YelpAPI(credentials['yelp']['api_key'])
        self.__data_path = "../data/raw"
        logger.info("initiation started.")

    def tw_verify_credentials(self):
        obj = self.twitter_api.VerifyCredentials()
        print(json.dumps(obj._json, indent=4, sort_keys=True))

    def tw_get_statuses(self, user_list):
        for username in user_list:
            with open(f'datasets/tw_{username}_statuses.json', 'w') as f:
                try:
                    f.write('{"statuses": [')
                    max_id = 0
                    while(True):
                        # status scheme available at: https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline.html
                        statuses = self.twitter_api.GetUserTimeline(
                            screen_name=username,
                            count=100,
                            max_id=max_id)

                        if len(statuses) == 1 and statuses[0].id == max_id:
                            break
                        else:
                            for status in statuses:
                                if status.id != max_id:
                                    f.write("%s," % json.dumps(status._json))

                            max_id = statuses[-1].id
                finally:
                    max_id != 0 and f.seek(f.tell() - 1, os.SEEK_SET)
                    f.write("]}")

    def tw_get_search(self, user_list):
        for user_name, keyword_list in user_list.items():
            with open(f'datasets/tw_{user_name}_searches.json', 'w') as f:
                try:
                    f.write('{"statuses": [')
                    max_id = 0
                    user = self.twitter_api.GetUser(screen_name=user_name)
                    keyword_list.append(f'{user.name}')
                    keyword_list.append(f'{user_name}')
                    keyword_list.append(f'#{user_name}')
                    keyword_list.append(f'@{user_name}')
                    term = '{}'.format(' OR '.join(keyword_list))
                    while(True):
                        # status scheme available at: https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets.html
                        statuses = self.twitter_api.GetSearch(
                            term=term.encode('utf-8'),
                            geocode=None,
                            count=100,
                            max_id=max_id)

                        if (len(statuses) == 1 and statuses[0].id == max_id) or statuses == []:
                            break
                        else:
                            for status in statuses:
                                if status.id != max_id:
                                    """status_text = json.dumps(status._json)
                                    status_json = json.loads(status_text)
                                    status_json['keyword'] = keyword"""
                                    f.write("%s," % json.dumps(status._json))
                            max_id = statuses[-1].id
                finally:
                    max_id != 0 and f.seek(f.tell() - 1, os.SEEK_SET)
                    f.write("]}")

    def tw_get_premium_search(self, keyword: str):
        with open(f'datasets/tw_{keyword.lower()}_searches_premium.json', 'w') as f:
            try:
                f.write('{"statuses": [')

                rule = gen_rule_payload(
                    pt_rule="near:\"New York, NY\" within:50mi".format(),
                    results_per_call=100,
                    from_date="2018-07-01",
                    to_date="2018-10-01"
                )

                rule = gen_rule_payload(
                    pt_rule="place:\"New York, NY\"".format(),
                    results_per_call=100,
                    from_date=(datetime.date.today() -
                               datetime.timedelta(31)).isoformat(),
                    to_date=datetime.date.today().isoformat()
                )

                next_token = None
                while True:
                    results = ResultStream(
                        rule_payload=rule,
                        **self.twitter_premium_api)
                    results.next_token = next_token

                    tweets = []

                    try:
                        tweets = list(results.stream())
                    except Exception as ex:
                        print(str(ex))

                    for tweet in tweets:
                        f.write("%s," % json.dumps(tweet))

                    if results.next_token is None:
                        break
                    else:
                        next_token = results.next_token

                next_token is not None and f.seek(f.tell() - 1, os.SEEK_SET)
                f.write("]}")

            except Exception as ex:
                print("Error:\n" + str(ex))

    def yp_get_businesses(self, business_list):
        """
        Get reviews for each business in the business_list and creates separate data files.
        File Type: JSON
        """
        for business in business_list:            
            with open(f'{self.data_path}/yp_{business}_competitors.json', 'w') as f:
                try:
                    f.write('{"businesses": [')
                    branch = self.yelp_api.business_query(business)
                    offset = 0
                    while(True):
                        try:
                            # status scheme available at: # https://www.yelp.com/developers/documentation/v3/business_search
                            competitors = self.yelp_api.search_query(
                                longitude=branch['coordinates']['longitude'],
                                latitude=branch['coordinates']['latitude'],
                                radius=40000,
                                # categories='bars,french'
                                sort_by='distance',
                                limit=50,
                                offset=offset)

                            f.write("%s," % json.dumps(
                                competitors['businesses']))
                            offset = offset + 50
                        except self.yelp_api.YelpAPIError:
                            break
                finally:
                    offset != 0 and f.seek(f.tell() - 1, os.SEEK_SET)
                    f.write("]}")

    def yp_get_competitors(self, business_list):
        """
        Gets business list in consideration to the existing business list file. Adds any additional business, if it is not recorded yet.
        """
        file_path = fp.yp_raw_competitors(self.data_path)
        index_list = []
        existing_list = []  
        """
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                current_file = f.readlines()
                if len(current_file) > 0:
                    existing_list = json.loads(current_file[0])
                    index_list = [_business["alias"] for _business in existing_list]
                    logger.info(f"existing file found: {len(index_list)} total entries")
        """
        with open(file_path, 'w') as f:
            # find businesses
            for business in business_list:
                new_list = []
                
                try:
                    logger.info(f"import started for : {business}")
                    branch = self.yelp_api.business_query(business)
                    offset = 0
                    while(True):
                        try:
                            # status scheme available at: # https://www.yelp.com/developers/documentation/v3/business_search
                            competitors = self.yelp_api.search_query(
                                longitude=branch['coordinates']['longitude'],
                                latitude=branch['coordinates']['latitude'],
                                radius=40000,
                                # categories='bars,french'
                                sort_by='distance',
                                limit=50,
                                offset=offset)
                            
                            # add alias name for distance measurement as dist_to_alias
                            businesses = competitors["businesses"]
                            [i.update({"dist_to_alias": business}) for i in businesses]  

                            for i in businesses:
                                if i['alias'] not in index_list:
                                    new_list.append(i)
                                    index_list.append(i['alias'])
                            
                            offset = offset + 50
                        except self.yelp_api.YelpAPIError:
                            break
                  
                finally:
                    existing_list.extend(new_list)
                    logger.info(f"import completed. existing: {len(existing_list)} new: {len(new_list)}")
            
            # saving into file
            json.dump(existing_list, f)
                    
    def yp_get_business_reviews(self, business_list):
        """
        Gets three reviews from the yelp api.
        """
        for business in business_list:
            with open(f'{self.data_path}/yp_{business}_rws.json', 'w') as f:
                try:
                    f.write('{"reviews": [')
                    offset = 0
                    while(True):
                        reviews_set = self.yelp_api.reviews_query(
                            business, limit=5, offset=offset)
                        reviews = reviews_set['reviews']
                        if len(reviews) > 0:
                            for review in reviews:
                                f.write("%s,\n" % review)

                            offset = offset + 5
                        else:
                            break
                finally:
                    offset != 0 and f.seek(f.tell() - 1, os.SEEK_SET)
                    f.write("]}")

    def yp_get_competitor_reviews(self, business_list=None, start_index=0, end_index=5):
        """
        Gets reviews by scraping through the site. Reviews are saved by business name and reviews. Uses Competitors reviews file as default file. Given index controls regions of Competitors. 
        business_list: None or List
        start_index: int, interested region's starting index
        end_index: int, interested region's ending index
        File Type: CSV
        """
        file_path = fp.yp_raw_competitors_reviews(self.data_path)          
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
        }
        columns = ['alias', 'ratingValue', 'dataPublished', 'description', 'author']
        df: pd.DataFrame
        # getting competitors list
        businesses_file_path = fp.yp_raw_competitors(self.data_path)
        businesses_index_list = []
        
        if os.path.exists(businesses_file_path):
            with open(businesses_file_path, 'r') as f:
                current_file = f.readlines()
                if len(current_file) > 0:
                    businesses_index_list = [_business["alias"] for _business in json.loads(current_file[0])]
        
        # needed every time
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                df = pd.read_csv(file_path)
                logger.info(f"existing file found. total reviews count: {len(df)}")

        # need only once, if file doesn't exists
        if os.path.exists(file_path) is False:
            with open(file_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                logger.info("file created at: {}".format(file_path))
        
        # ops           
        with open(file_path, 'a', newline='') as f:    
            if business_list is None:                
                business_list = businesses_index_list

            current_index = start_index - 1
            for business in business_list[start_index: end_index]: 
                cnt_imported = 0 
                current_index = current_index + 1
                logger.info(f"index: {current_index} of {end_index - 1}")
                try:
                    writer = csv.writer(f)                    
                    logger.info(f"import started for : {business}")
                    start = 0
                    cnt_requests = 0
                    while (True):
                        url = '{}/{}?sort_by=date_desc&start={}'.format('https://www.yelp.com/biz', business, start)
                        response = requests.get(url, headers)

                        soup = BeautifulSoup(response.text, 'html.parser')
                        html_script = soup.findAll('script', {'type': 'application/ld+json'})[-1]
                        obj = json.loads(html_script.string)

                        reviews = obj['review']
                        if len(reviews) > 0:
                            for review in reviews:
                                data = [
                                    business,
                                    review['reviewRating']['ratingValue'],
                                    review['datePublished'],
                                    review['description'],
                                    review['author']
                                ]

                                check = np.array(data, dtype='O')
                                if not (df.values == check).all(1).any():
                                    writer.writerow(data)
                                    cnt_imported = cnt_imported + 1

                            start = start + 20
                            cnt_requests = cnt_requests + 1
                        else:
                            logger.info(f"import completed. total reviews cnt: {cnt_imported} total request cnt: {cnt_requests}")
                            break
                except Exception as ex:
                    logger.warning(f"error: alias: {business} index: {current_index} total reviews cnt: {cnt_imported}")
                    logger.warning(f"error message: {ex}")
                    logger.warning("Let me sleep for some time..")
                    second = int(round(random.expovariate(1) * 100))
                    time.sleep(second)
                    logger.warning(f"{second} seconds slept, now back on scrapping..")
                    continue

    def yp_get_business_reviews2(self, business_list):
        """
        Gets reviews by scraping through the site.
        """
        for business in business_list:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

            with open(f'{self.data_path}/yp_{business}_rws.json', 'w') as f:
                try:
                    f.write('{"reviews": [')
                    start = 0
                    while (True):
                        url = '{}/{}?sort_by=date_desc&start={}'.format(
                            'https://www.yelp.com/biz', business, start)
                        response = requests.get(url, headers)

                        soup = BeautifulSoup(response.text, 'html.parser')
                        html_script = soup.find(
                            'script', {'type': 'application/ld+json'})
                        obj = json.loads(html_script.string)

                        reviews = obj['review']
                        if len(reviews) > 0:
                            for review in reviews:
                                data = {
                                    'ratingValue': review['reviewRating']['ratingValue'],
                                    'datePublished': review['datePublished'],
                                    'description': review['description'],
                                    'author': review['author']
                                }
                                f.write("%s," % json.dumps(data))
                            start = start + 20
                        else:
                            break
                finally:
                    start != 0 and f.seek(f.tell() - 1, os.SEEK_SET)
                    f.write("]}")

            with open(f'datasets/yp_businesses.json', 'a') as f:
                obj['review'] = []