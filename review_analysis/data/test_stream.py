import requests
import json
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
}

url = '{}/{}?sort_by=date_desc&start={}'.format(
    'https://www.yelp.com/biz', 'kimos-maui-lahaina', 0
)
with requests.get(url, headers, stream=True) as response:
    soup = BeautifulSoup(response.text, 'html.parser')
    html_script = soup.findAll('script', {'type': 'application/ld+json'})[-1]
    obj = json.loads(html_script.string)

    reviews = obj['review']

    print(reviews)