from __future__ import unicode_literals
import re
from nltk.corpus import stopwords
import spacy, nltk
from nltk.stem import WordNetLemmatizer
import en_core_web_sm

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

nlp = en_core_web_sm.load()

try:
    stop_words_nltk = stopwords.words('english')
except Exception:
    nltk.download('stopwords')
    stop_words_nltk = stopwords.words('english')

stop_words_spacy = list(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere n't

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves

'd 'll 'm 're 's 've
""".split()
)

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


def get_filtered_pos(sentence, filter_pos=['ADJ', 'VERB', 'NOUN', 'PROPN', 'FW']):
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
                if word[1] in filter_pos and word[0] != 'XnumberX':
                    sentence_pos.append(word)
            text_pos.append(sentence_pos)
        return text_pos
    return sentence
