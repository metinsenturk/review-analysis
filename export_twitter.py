import utilities
import twitter


class export_data:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.credentials = utilities.get_credidentials()
        self.twitter_api = twitter.Api(consumer_key=self.credentials['twitter']['consumer_key'],
                      consumer_secret=self.credentials['twitter']['consumer_secret'],
                      access_token_key=self.credentials['twitter']['access_token_key'],
                      access_token_secret=self.credentials['twitter']['access_token_secret'])

    def verify_twitter(self):
        statuses = self.twitter_api.GetUserTimeline(screen_name='KimosRestaurant',count=5, max_id=0)
        for s in statuses:
            print(f"{s.id}: {s.created_at} - {s.text}")
    
    def get_statuses(self):
        max_id = 0
        statuses = []
        
        with open('twitter_kimosrestaurant.json', 'w') as f:
            while(True):
                # status scheme available at: https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline.html
                statuses = self.twitter_api.GetUserTimeline(screen_name='KimosRestaurant',count=5, max_id=max_id)
                
                for status in statuses:
                    f.write("%s\n" % status)
                
                max_id = statuses[-1].id
        

if __name__ == '__main__':
    a = export_data()
    # a.verify_twitter()
    a.get_statuses()