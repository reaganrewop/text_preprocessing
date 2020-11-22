import logging
import re
import itertools
import string
import os
import nltk
import nltk.data
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from .util import (
    get_stop_words,
    get_contraction_mapping,
    get_punct,
    get_web_url_regex,
    get_grammer,
    leaves,
    acceptable_word,
    get_terms
)
from typing import List, Tuple, Union
logger = logging.getLogger(__name__)

sent_detector = nltk.data.load("tokenizers/punkt/PY3/english.pickle")


class Preprocessor:
    original_text = []
    stop_words = []
    preprocessed_text = []
    pos_tagged_sents = []
    web_url_regex = None
    grammer = None

    def __init__(self):
        self.stop_words = get_stop_words()
        self.contraction_mapping = get_contraction_mapping()
        self.punct = get_punct()
        self.web_url_regex = get_web_url_regex()
        self.grammer = get_grammer()
        self.chunker = nltk.RegexpParser(self.grammer)

    def get_tokenized_sent(self, para_text) -> List[str]:
        sentences = sent_detector.tokenize(para_text.strip())
        return sentences

    def expand_contractions(self, sentence) -> str:
        """
        Description : Expand contractions like "y'all" to "you all" using known mapping.
        input : A single sentence as a string.
        output : A expanded contraction sentence as string
        """
        # Note: shouldn't do nltk.word_tokenize, as it splits it's as "it" and "'s"
        words = sentence.split(" ")
        if words:
            for word in words:
                if self.contraction_mapping.get(word.lower()):
                    sentence = sentence.replace(
                        word, self.contraction_mapping[word.lower()]
                    )
        return sentence.strip()

    def evaluate_sentence_quality(self, mod_sent) -> str:
        """
        Description : evalute the quality of the text.
        input : A pos-tagged sentence
        output : str which has (strong/weak)
        """

        tree = self.chunker.parse(mod_sent)
        terms = get_terms(tree)
        terms_list = [t for t in terms]
        # if len(terms_list) >= 3:
        if True in [True if len(t) >= 2 else False for t in terms_list]:
            validity = "strong"

        else:
            validity = "weak"
        return validity

    def unkown_punct(self, sentence, remove_punct) -> str:
        """
        Description : remove punctuation.
        input : A single sentence as a string.
        output : A string.
        """
        sentence = sentence.replace("\n", ". ")
        for p in self.punct:
            if p in sentence:
                if remove_punct:
                    sentence = sentence.replace(" " + p + " ", " ")
                    sentence = sentence.replace("" + p + " ", " ")
                    sentence = sentence.replace(" " + p + "", " ")
                    # shouldn't remove "high-class" kind of tokens
                    sentence = re.sub(r'['+ p  + ']' + '['+ p  + ']+', '', sentence)
        # removes full stop
        sentence = re.sub(r'[.]+', '', sentence)
        return sentence

    def remove_number(self, sentence) -> str:
        """
        Description : replace numbers with XnumberX.
        input : A single sentence as a string.
        output : A string.
        """
        sentence = re.sub(r"\d+\.+\d+", "__NUMBER__", " " + sentence + " ")
        sentence = re.sub(r"\d+", "__NUMBER__", " " + sentence + " ")
        return sentence[2:-2]

    def remove_stopwords(self, sentence) -> str:
        """
        Description : remove stopwords.
        input : A single sentence as a string.
        output : A string without stopwords.
        """
        words = nltk.word_tokenize(sentence)
        if words:
            for word in words:
                if word.lower() in self.stop_words:
                    sentence = sentence.replace(" " + word.lower() + " ", " ")
                    if word.lower() == sentence[:len(word.lower())]:
                        sentence = sentence[len(word.lower()):]
                    if word.lower() == sentence[-len(word.lower()):]:
                        sentence = sentence[:-len(word.lower())]
        return sentence.strip()

    def lemmatization(self, sentence) -> str:
        """
        Description : do Lemmatization.
        input : A single sentence as a string.
        output : A string with words lemmatized.
        """
        lemmatizer = WordNetLemmatizer()
        words = sentence.split(" ")
        if words:
            for word in words:
                if word == lemmatizer.lemmatize(word):
                    sentence = sentence.replace(
                        word, lemmatizer.lemmatize(word)
                    )
        return sentence

    def get_pos(self, sentence) -> List[Tuple[str, str]]:
        """
        Description : get pos tags of word tokenized sentence
        input : A single string sentence.
        output : A list of sentence where each sentence contains a list of tuple with word, POS.
        """
        sentence_pos = []
        filtered_sentence = sentence.replace(".", ". ")

        tokenized_text = word_tokenize(filtered_sentence)
        pos_tags = nltk.pos_tag(tokenized_text)
        for tags in pos_tags:
            sentence_pos.append(tags)
        return sentence_pos

    def clean_text(self, text, remove_punct=False, mask_numbers=False):
        """
        Description : Clean the text with common preprocessing rules.
        input : A single string sentence.
        output : Cleaned string of text.
        """
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r"", text)
        text = self.expand_contractions(text)
        if mask_numbers:
            text = self.remove_number(text)
        text = re.sub(self.web_url_regex, "__url__", text)
        #specific
        if '__url__' in text:
            return ''
        text = text.replace('__url__', '').replace('__NUMBER__', '')
        text = self.unkown_punct(text, remove_punct)
        text = (
            text.replace("\\n", "")
            .replace("’", "'")
            .replace("\\", "")
            .replace("‘", "'")
        )
        # changes: removed "replace(":", ". ")"
        text = (
            text.replace("”", "'")
            .replace("“", "'")
        )
        text = text.replace("\u200a—\u200a", " ").replace("\xa0", "")
        text = re.sub(" +", " ", text)
        text = text.replace("\t", "")
        text = text.replace("\n", "")
        return text.strip()

    def get_preprocessed_text(
        self,
        text,
        lemma=False,
        stop_words=False,
        word_tokenize=False,
        remove_punct=False,
        mask_numbers=False,
        pos=False,
        sentence_quality=False
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """
        Description: Does all the basic pre-processing.
        Input: Set of sentence(s) as a string or list of sentence(s).
        Output: List of pre-processed sentences.
        """
        if isinstance(text, str):
            sentences = self.get_tokenized_sent(text)
        elif isinstance(text, list):
            sentences = text
        else:
            raise Exception(
                "Unknown type of input. Please, pass a string or list of sentences."
            )

        preprocessed_text = []
        pos_tagged_sents = []
        for index, sent in enumerate(sentences):
            mod_sent = []
            mod_sent_post = []
            mod_sent = self.clean_text(sent, remove_punct, mask_numbers)
            if mod_sent != "":
                if pos:
                    mod_sent_pos = mod_sent[:]
                    mod_sent_pos = self.get_pos(mod_sent_pos)
                    pos_tagged_sents.append(mod_sent_pos)

                if sentence_quality:
                    if not pos:
                        mod_sent_pos = mod_sent[:]
                        mod_sent_pos = self.get_pos(mod_sent_pos)

                    strength = self.evaluate_sentence_quality(mod_sent_pos)

                    if strength == "weak":
                        continue

                if stop_words:
                    mod_sent = self.remove_stopwords(mod_sent)

                if lemma:
                    mod_sent = self.lemmatization(mod_sent)

                if word_tokenize:
                    mod_sent = nltk.word_tokenize(mod_sent)

                preprocessed_text.append(mod_sent.strip())

        self.preprocessed_text = preprocessed_text

        if pos:
            self.pos_tagged_sents = pos_tagged_sents
            return preprocessed_text, pos_tagged_sents

        return preprocessed_text
