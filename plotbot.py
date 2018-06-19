'''Script to perform sentiment analysis on a tweet
'''

import re
import time

# import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd

# Import tweepy and the API keys, secrets, ...
import tweepy
from config import (
    CONSUMER_KEY,
    CONSUMER_SECRET,
    ACCESS_TOKEN,
    ACCESS_TOKEN_SECRET
)

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ANALYZER = SentimentIntensityAnalyzer()

# Setup Tweepy API Authentication
AUTH = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
AUTH.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
API = tweepy.API(AUTH, parser=tweepy.parsers.JSONParser())

USERNAME = API.me()['screen_name']
PNG_FILE_NAME = 'tweet_polarity.png'
ALREADY_TREATED = []


def get_mentions():
    '''Search for new mentions

    Returns:
        list of tuples, each tuple contains the screen name of the
            user to analyze and the screen name of the user who
            requested the analysis.
    '''
    results = API.search(f"@{USERNAME} analyze:")
    mentions = []

    for tweet in results['statuses']:
        tweet_id = tweet['id']
        if tweet_id not in ALREADY_TREATED:
            tweet_text = tweet['text']
            # Get the screen name of user who requested the analysis
            tweeter = tweet['user']['screen_name']
            # Use regular expression to find the screen name to analyze
            pattern = re.compile(r'@' + USERNAME.lower() +
                                 r' analyze:\s*(@?[A-Za-z][A-Za-z0-9-_]+)',
                                 re.IGNORECASE)
            match = re.match(pattern, tweet_text)
            if match:
                mention = match.group(1)
                mentions.append((mention, tweeter))
                ALREADY_TREATED.append(tweet_id)

    return mentions


def sentiment_analysis(screen_name):
    '''Performs a sentiment analysis on latest tweets from a given user.

    Arguments:
        - screen_name (str): the screen name of the user to analyze
            (e.g. '@DalaiLama')

    Returns:
        tuple with:
            - file name (.png) with the sentiment analysis plot
            - id of the most negative tweet
            - id of the most positive tweet
    '''

    # Create a generic dictionary for holding all tweet information
    tweet_data = {
        'ID': [],
        'Screen Name': [],
        'Text': [],
        'Date': [],
        'Compound': [],
        'Negative': [],
        'Positive': [],
        'Neutral': []
    }

    # Grab 500 tweets from the target source
    for page in range(1, 6):

        # Grab the tweets
        tweets = API.user_timeline(screen_name, page=page, count=100)

        # For each tweet store it into the dictionary
        for tweet in tweets:

            # Run sentiment analysis on each tweet using Vader
            sentiment = ANALYZER.polarity_scores(tweet['text'])

            if sentiment['compound'] != 0.0 and sentiment['neu'] != 1.0:
                # All data is grabbed from the JSON returned by Twitter
                tweet_data['ID'].append(tweet['id'])
                tweet_data['Screen Name'].append(tweet['user']['name'])
                tweet_data['Text'].append(tweet['text'])
                tweet_data['Date'].append(tweet['created_at'])
                tweet_data['Compound'].append(sentiment['compound'])
                tweet_data['Positive'].append(sentiment['pos'])
                tweet_data['Neutral'].append(sentiment['neu'])
                tweet_data['Negative'].append(sentiment['neg'])

    # Create a dataframe
    tweet_df = pd.DataFrame(tweet_data)

    # Exit function if there is no result to plot
    if tweet_df.empty:
        print('There was no tweet to analyze')
        return None, None, None

    # Convert dates (currently strings) into datetimes
    tweet_df['Date'] = pd.to_datetime(tweet_df['Date'])

    # Sort the dataframe by date
    tweet_df.sort_values('Date', inplace=True)
    # tweet_df.set_index('Date', drop=True, inplace=True)

    # Print best and worst tweets
    print(screen_name)
    print('\nMost negative tweet:')
    neg_id = tweet_df.loc[tweet_df['Compound'].idxmin(), 'ID']
    print(f'https://twitter.com/statuses/{neg_id}')
    print('\nMost positive tweet:')
    pos_id = tweet_df.loc[tweet_df['Compound'].idxmax(), 'ID']
    print(f'https://twitter.com/statuses/{pos_id}')

    # Plot the data
    fig = plt.figure(figsize=(12, 5))
    plt.xticks(rotation=45, ha='right')
    plt.plot(tweet_df['Date'], tweet_df['Compound'], marker='o',
             color='steelblue', alpha=0.5, linewidth=1)

    # Add plot labels and title
    plt.xlabel('Date of tweet')
    plt.ylabel('Tweet polarity')
    plt.title(f'Sentiment Analysis of Tweets for {screen_name}')
    plt.tight_layout()

    # Add line for mean compound score
    xmin, xmax = plt.xlim()
    mean_score = tweet_df['Compound'].mean()
    plt.hlines(mean_score, xmin, xmax, colors='red',
               linestyles='-.', label=f'Avg: {mean_score:.2f}')
    plt.legend(loc='lower right')
    fig.savefig(PNG_FILE_NAME)

    return PNG_FILE_NAME, neg_id, pos_id


def send_tweet(mention, file_name=None, neg_id=None, pos_id=None):
    '''Sends a tweet with the sentiment analysis plot

    Arguments:
        mention (str): screen name of user to mention in the tweet.

    Keyword arguments:
        file_name (default=None): name of the image file to tweet.
        neg_id (default=None): id of the most negative tweet.
        pos_id (default=None): id of the most positive tweet.
    '''
    if file_name is None:
        message = (f'Sorry @{mention}, there was no data associated '
                   f'with that user name')
        API.update_status(message)
    else:
        message = (f'@{mention}! '
                   f'Here\'s the analysis you requested\n'
                   f'Most negative tweet: '
                   f'https://twitter.com/statuses/{neg_id}\n'
                   f'Most positive tweet: '
                   f'https://twitter.com/statuses/{pos_id}\n')
        API.update_with_media(file_name, message)


if __name__ == '__main__':

    # Run until the end of times
    while True:
        # Find if anyone requested a sentiment analysis
        MENTIONS = get_mentions()

        # Loop through all new requests
        for NAME_TO_ANALYZE, TWEETERNAME in MENTIONS:
            # Run the sentiment analysis
            FILENAME, NEG_ID, POS_ID = sentiment_analysis(NAME_TO_ANALYZE)
            # Tweet the results
            send_tweet(TWEETERNAME, file_name=FILENAME, neg_id=NEG_ID,
                       pos_id=POS_ID)

        # Sleep for a little while
        time.sleep(60)
