import utilities
import twitter
from yelpapi import YelpAPI
import facebook

class export_data:
    def __init__(self):
        credentials = utilities.get_credidentials()
        self.twitter_api = twitter.Api(consumer_key=credentials['twitter']['consumer_key'],
                      consumer_secret=credentials['twitter']['consumer_secret'],
                      access_token_key=credentials['twitter']['access_token_key'],
                      access_token_secret=credentials['twitter']['access_token_secret'])
                      
        self.yelp_api = YelpAPI(credentials['yelp']['api_key'])
        self.facebook_api = facebook.GraphAPI(access_token=credentials['facebook']['page_access_token'])
        self.facebook_api.get_object

    def verify_twitter(self):
        print(self.twitter_api.VerifyCredentials())
    
    def tw_get_statuses(self, user_list):
        for username in user_list:
            with open(f'datasets/tw_{username}.json', 'w') as f:
                max_id = 0
                while(True):
                    # status scheme available at: https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline.html
                    statuses = self.twitter_api.GetUserTimeline(screen_name=username,count=5, max_id=max_id)
                    
                    if len(statuses) > 0:
                        for status in statuses:
                            f.write("%s\n" % status)
                    
                        max_id = statuses[-1].id
                    else:
                        break
    
    def yp_get_business(self, business_list):
        with open('datasets/yp_businesses.json', 'w') as f:
            for business in business_list:
                response = self.yelp_api.business_query(business)
                f.write("%s\n" % response)

    def yp_get_business_reviews(self, business_list):
        for business in business_list:
            with open(f'datasets/yp_{business}_rw.json', 'w') as f:
                offset = 0
                while(True):
                    reviews_set = self.yelp_api.reviews_query(business, limit=5, offset=offset)
                    reviews = reviews_set['reviews']
                    if len(reviews) > 0:
                        for review in reviews:
                            f.write("%s\n" % review)
                        
                        offset = offset + 5
                    else:
                        break

    def fb_(self):
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
        _arg = {'fields': ','.join(metrics_page_impression), }
        friends = self.facebook_api.get_object(id='789708227817519', args=_arg)
        friends = self.facebook_api.get_connections(id='me', connection_name='events', args=_arg)
        # friends = self.facebook_api.get_connections(id='789708227817519', connection_name='insights', args=_arg)
        print(friends)


if __name__ == '__main__':
    a = export_data()
    # a.verify_twitter()
    # a.tw_get_statuses(['KimosRestaurant',])
    # a.yp_get_business(['kimos-maui-lahaina'])
    # a.yp_get_business_reviews(['kimos-maui-lahaina'])
    a.fb_()