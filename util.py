import re
from nltk.corpus import stopwords
import spacy, nltk
from nltk.stem import WordNetLemmatizer


try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")
stop_words_spacy = list(nlp.Defaults.stop_words)
stop_words_nltk = stopwords.words('english')

stop_words = list(set(stop_words_nltk + stop_words_spacy))


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
punct = "/-'?!,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


def expand_contractions(sentence):
    '''
    Description : Expand contractions like "y'all" to "you all" using known mapping.
    input : A single sentence as a string.
    output : A expanded contraction sentence as string
    '''
    words = sentence.split(' ')
    if words:
        for word in words:
            if not not contraction_mapping.get(word):
                sentence = sentence.replace(word, contraction_mapping[word])
    return sentence


def unkown_punct(sentence, remove_punct):
    '''
    Description : remove punctuation.
    input : A single sentence as a string.
    output : A string.
    '''
    for p in punct:
        if p in sentence:
            if not remove_punct and p not in {',', '.', '?'}:
                sentence = sentence.replace(p, '')
    return sentence


def remove_number(sentence):
    '''
    Description : replace numbers with XnumberX.
    input : A single sentence as a string.
    output : A string.
    '''
    sentence = re.sub("\ \d+\ ", " XnumberX ", " " + sentence + " ")
    sentence = re.sub("\ \d+\ ", " XnumberX ", " " + sentence + " ")
    return sentence[2:-2]


def remove_stopwords(sentence):
    '''
    Description : remove stopwords.
    input : A single sentence as a string.
    output : A string without stopwords.
    '''
    words = sentence.split(' ')
    if words:
        for word in words:
            if word in stop_words:
                sentence = sentence.replace(" " + word + " ", " ")
    return sentence


def lemmatization(sentence):
    '''
    Description : do Lemmatization.
    input : A single sentence as a string.
    output : A string with words lemmatized.
    '''
    lemmatizer = WordNetLemmatizer()
    words = sentence.split(' ')
    if words:
        for word in words:
            if word == lemmatizer.lemmatize(word):
                sentence = sentence.replace(word, lemmatizer.lemmatize(word))
    return sentence


def get_pos(sentence):
    '''
    Description : get pos tags of word tokenized sentence
    input : A single string sentence.
    output : A list of sentence where each sentence contains a list of tuple with word, POS.
    '''
    sentence_pos = []
    doc = nlp(sentence)

    for token in doc:
        sentence_pos.append((token.text, token.pos_))
    return sentence_pos


def get_filtered_pos(sentence, filter_pos=['ADJ', 'VERB', 'NOUN', 'PROPN', 'FW']]):
    '''
    Description : Filter POS with respect to the given list.
    Input : A list of sentence, where sentence consist of \
            list of tuple with word,pos tag.
    Output : return the filtered POS tag for pos tags\
            which are present in filter_pos
    '''
    text_pos = []
    if filter_pos:
        for sent in sentence:
            sentence_pos = []
            for word in sent:
                if word[1] in filter_pos:
                    sentence_pos.append(word)
            text_pos.append(sentence_pos)
        return text_pos
    return sentence
