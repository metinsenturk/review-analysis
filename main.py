import utilities
import twitter
from yelpapi import YelpAPI
import facebook
from bs4 import BeautifulSoup
import requests
import os
import json
import datetime

class main:
    def __init__(self):
        credentials = utilities.get_credidentials()
        self.twitter_api = twitter.Api(
            consumer_key=credentials['twitter']['consumer_key'],
            consumer_secret=credentials['twitter']['consumer_secret'],
            access_token_key=credentials['twitter']['access_token_key'],
            access_token_secret=credentials['twitter']['access_token_secret'])
        self.yelp_api = YelpAPI(credentials['yelp']['api_key'])
        self.facebook_api = facebook.GraphAPI(access_token=credentials['facebook']['page_access_token'])
           
    def tw_verify_credentials(self):
        print(self.twitter_api.VerifyCredentials())
    
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

    def yp_get_business(self, business_list):
        for business in business_list:
            with open(f'datasets/yp_{business}_competitors.json', 'w') as f:
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

                            f.write("%s," % json.dumps(competitors['businesses']))
                            offset = offset + 50
                        except self.yelp_api.YelpAPIError:
                            break
                finally:
                    offset != 0 and f.seek(f.tell() - 1, os.SEEK_SET)
                    f.write("]}")            

    def yp_get_business_reviews(self, business_list):
        for business in business_list:
            with open(f'datasets/yp_{business}_rws.json', 'w') as f:
                try:
                    f.write('{"reviews": [')
                    offset = 0
                    while(True):
                        reviews_set = self.yelp_api.reviews_query(business, limit=5, offset=offset)
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

    def yp_get_business_reviews2(self, business_list):
        for business in business_list:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

            with open(f'datasets/yp_{business}_rws.json', 'w') as f:
                try:
                    f.write('{"reviews": [')
                    start = 0  
                    while (True):  
                        url = '{}/{}?start={}'.format('https://www.yelp.com/biz', business, start)
                        response = requests.get(url, headers)
                        
                        soup = BeautifulSoup(response.text, 'html.parser')
                        html_script = soup.find('script', {'type': 'application/ld+json'})
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
                f.write("%s" % json.dumps(obj))

    def fb_(self):
        """
        Incomplete.
        """        
        metric_open_graph_rating = [
            'created_time',
            'has_rating',
            'has_review',
            'open_graph_story',
            'rating',
            'review_text',
            'reviewer'
        ]
        metrics = [
            'page_content_activity_by_age_gender_unique',
            'page_content_activity_by_city_unique',
            'page_content_activity_by_country_unique',
            'page_content_activity_by_locale_unique',
            'page_content_activity',
            'page_content_activity_by_action_type',
            'post_activity',
            'post_activity_unique',
            'post_activity_by_action_type',
            'post_activity_by_action_type_unique'
        ]
        metrics_page_impression = [
            'page_impressions',
            'page_impressions_unique',
            'page_impressions_paid',
            'page_impressions_paid_unique',
            'page_impressions_organic',
            'page_impressions_organic_unique',
            'page_impressions_viral',
            'page_impressions_viral_unique'
        ]
        _arg = {'fields': ','.join(metric_open_graph_rating)}
        friends = self.facebook_api.get_object(id='789708227817519', args=_arg)
        friends = self.facebook_api.get_connections(id='me', connection_name='likes', args=_arg)
        # friends = self.facebook_api.get_connections(id='789708227817519', connection_name='insights', args=_arg)
        print(friends)


if __name__ == '__main__':
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
    twitter_user_searches = {
        'KimosRestaurant': [],
        'JakesInDelMar': [],
        'SunnysideResort': [],
        'dukeshb': [],
        'DukesLaJolla': [],
        'DukesMalibu': [],
        'DukesBeachHouse': [],
        'DukesInKauai': [],
        'DukesWaikiki': [],
        'hulagrillwaiks': [],
        'HulaGrillMaui': [],
        'KeokisParadise': [],
        'LeilanisMaui': []
    }
    yelp_branches = [
        'kimos-maui-lahaina', 
        'sunnyside-tahoe-city-2', 
        'dukes-huntington-beach-huntington-beach-2', 
        'dukes-la-jolla-la-jolla',
        'dukes-malibu-malibu-2',
        'dukes-beach-house-lahaina',
        'dukes-kauai-lihue-3',
        'dukes-waikiki-honolulu-2',
        'hula-grill-waikiki-honolulu-3',
        'hula-grill-kaanapali-lahaina-2',
        'keokis-paradise-koloa',
        'leilanis-lahaina-2'
    ]
    a = main()
    # a.verify_twitter()
    # a.tw_get_statuses(twitter_users)
    a.tw_get_search(twitter_user_searches)
    # a.yp_get_business_reviews2(yelp_branches)
    # a.fb_()