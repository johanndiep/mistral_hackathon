import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


def extract_nouns(query):
    # Download necessary NLTK data files
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

    # Define stopwords and tokenize the query
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(query)

    # POS tagging
    tagged_words = pos_tag(word_tokens)

    # Extract nouns: remove stopwords and non-alphabetic tokens, and filter for nouns
    nouns = [
        word
        for word, pos in tagged_words
        if word.isalpha() and word.lower() not in stop_words and pos.startswith("NN")
    ]

    # Return nouns as a comma-separated string
    return nouns
