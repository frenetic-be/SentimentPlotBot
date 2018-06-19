'''Script to retrieve twitter credentials for sentiment analysis
'''
import os

# Twitter credentials
CONSUMER_KEY = os.environ.get('TWITTER_SENTIMENT_KEY')
CONSUMER_SECRET = os.environ.get('TWITTER_SENTIMENT_KEY_SECRET')
ACCESS_TOKEN = os.environ.get('TWITTER_SENTIMENT_TOKEN')
ACCESS_TOKEN_SECRET = os.environ.get('TWITTER_SENTIMENT_TOKEN_SECRET')
