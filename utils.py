import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def extract_keywords(query):
    # Download necessary NLTK data files
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    # Define stopwords and tokenize the query
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(query)

    # Extract keywords: remove stopwords and non-alphabetic tokens
    keywords = [
        word
        for word in word_tokens
        if word.isalpha() and word.lower() not in stop_words
    ]

    # Return keywords as a comma-separated string
    return ", ".join(keywords)
