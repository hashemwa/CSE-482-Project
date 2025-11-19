import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import nltk

nltk.download("punkt_tab")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from collections import Counter
from nltk.util import bigrams
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from string import punctuation

df = pd.read_csv(
    "twitter_training.csv", names=["tweet_id", "subject", "sentiment", "review_text"]
)

## Cleaning
ENGLISH_STOPWORDS_SET = set(stopwords.words("english"))


def remove_stopwords_from_list(token_list):
    """
    Filters a list of tokens, returning only those not in the stopword set.
    """
    # This list comprehension is the efficient way to filter tokens
    return [word for word in token_list if word not in ENGLISH_STOPWORDS_SET]


def remove_punctuation_from_list(token_list):
    """
    Filters a list of tokens, returning only those not in the stopword set.
    """
    # This list comprehension is the efficient way to filter tokens
    return [word for word in token_list if word not in punctuation]


df = df[df["sentiment"].isin(["Positive", "Negative"])]

df["review_text"] = df["review_text"].str.lower()

df = df.dropna(subset=["review_text"])

# Remove whitespace
df["cleaned_text"] = df["review_text"].str.replace(r"\s+", " ", regex=True)

df["tokens"] = df["cleaned_text"].apply(word_tokenize)

# remove punctuation
df["tokens"] = df["tokens"].apply(remove_punctuation_from_list)

df["tokens"] = df["tokens"].apply(remove_stopwords_from_list)

df = df.drop_duplicates(subset=["review_text", "tokens"])
