import os
import nltk
import nltk.data


if os.path.isdir("/tmp/nltk_data"):
    nltk.data.path.append("/tmp/nltk_data")
    try:
        nltk.data.find("wordnet")
        nltk.data.find("stopwords")
        nltk.data.find("tokenizers/punkt/PY3/english.pickle")

    except LookupError:
        nltk.download("wordnet", download_dir="/tmp/nltk_data")
        nltk.download("stopwords", download_dir="/tmp/nltk_data")
        nltk.download(
            "averaged_perceptron_tagger", download_dir="/tmp/nltk_data"
        )
        nltk.download("punkt", download_dir="/tmp/nltk_data")


else:
    nltk.download("wordnet", download_dir="/tmp/nltk_data")
    nltk.download("stopwords", download_dir="/tmp/nltk_data")
    nltk.download(
        "averaged_perceptron_tagger", download_dir="/tmp/nltk_data"
    )
    nltk.download("punkt", download_dir="/tmp/nltk_data")
    nltk.data.path.append("/tmp/nltk_data")
from .preprocess import Preprocessor
