print('Importing packages...')
import numpy as np
from collections import defaultdict
from csv import DictReader
from preprocessing import preprocess_tweets
from tweet import Tweet
from progress.bar import ShadyBar

# static variables
LABELS = ('pos', 'neg', 'com', 'obj')
CSV_LABELS = ('positive', 'negative', 'complicated')
HAS_SENTIMENT_KEY = 'does_the_author_express_sentiment_in_this_tweet_'
POS_NEG_COM_KEY = 'is_the_sentiment_expressed_positive_or_negative_'
IS_GOLD_KEY = '_golden'
ID_KEY = '_unit_id'
TWEET_ID = 'tweet_id'

# loading functions
def load_tweets_from_csv(fname='../data/annotated_tweets.csv',
                         preprocess=True,
                         serialize=True):
    # Load the data into memory
    print('Loading MTSA csv...')
    ids_to_content = defaultdict(lambda: [])
    with open(fname) as f:
        csv_reader = DictReader(f)
        for i, row in enumerate(csv_reader):
            ids_to_content[row[ID_KEY]].append(row)

    # construct the tweets and labels
    bar = ShadyBar('Labelling MTSA', max=len(ids_to_content))
    tweets = []
    for sample in ids_to_content.values():
        bar.next()
        csv_twt = sample[0]

        # skip the test questions used for crowdflower!
        if csv_twt[IS_GOLD_KEY] == 'true':
            continue

        # build up the tweet statistics of labels
        tweet_stats = {s: 0 for s in LABELS}
        for labelling in sample:
            if labelling[HAS_SENTIMENT_KEY] == 'no':
                tweet_stats['obj'] += 1
            for key in CSV_LABELS:
                if labelling[POS_NEG_COM_KEY] == key:
                    tweet_stats[key[0:3]] += 1

        # Skipping tweet that had < 5 annotations
        if sum(tweet_stats.values()) < 5:
            continue

        # extract the necessary data
        tweet = Tweet(csv_twt[TWEET_ID], csv_twt['text'], csv_twt['topic'])
        tweet.labelling = tweet_stats
        tweets.append(tweet)
    bar.finish()

    """
    The preprocessing pipeline is: (see preprocessing.py)
        (tokenize),
        (filter_tokens),
        (remove_mentions),
        (split_hashtags),
        (autocorrect),
        (lemmatize)
    """
    print('Removed {} tweets that had < 5 annotations.'.format(len(ids_to_content) - len(tweets)))
    print('We now have a total of {} tweets in the MTSA.'.format(len(tweets)))
    if preprocess:
        preprocess_tweets(tweets)

    # save data if desired
    if serialize:
        np.save('../data/processed_annotated_tweets.npy', np.array(tweets))
    return tweets

def load_tweets_from_npy(fname='../data/processed_annotated_tweets.npy'):
    print('Loading npy pickle...')
    return np.load(fname)

if __name__ == '__main__':
    load_tweets_from_csv()
    tweets = load_tweets_from_npy()
    print(tweets[0])
