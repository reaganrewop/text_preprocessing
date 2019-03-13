import nltk
import nltk.data
from text_preprocessing.util import expand_contractions, unkown_punct, \
    remove_number, remove_stopwords, lemmatization, get_pos, get_filtered_pos


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def preprocess(text, lemma=False, stop_words=True, word_tokenize=False, \
               remove_punct=True, pos=False):
    '''
    Description: Does all the basic pre-processing.
    Input: Set of sentence(s) as a string.
    Output: List of pre-processed sentences.
    '''
    pos_text = []
    if text:
        sentence = sent_detector.tokenize(text.strip())
        for index, sent in enumerate(sentence):
            mod_sent = expand_contractions(sent)
            mod_sent = unkown_punct(mod_sent, remove_punct)
            mod_sent = remove_number(mod_sent)
            if stop_words:
                mod_sent = remove_stopwords(mod_sent)
            if lemma:
                mod_sent = lemmatization(mod_sent)
            if pos:
                mod_sent_pos = mod_sent[:]
                mod_sent_pos = get_pos(mod_sent_pos)
                pos_text.append(mod_sent_pos)
            if word_tokenize:
                mod_sent = nltk.word_tokenize(mod_sent)
            sentence.remove(sent)
            sentence.insert(index, mod_sent)
        if pos:
            return sentence, pos_text
        return sentence
