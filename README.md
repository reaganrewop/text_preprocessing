# Text_Preprocessing
## A light-weight text preprocessing package which does what is most needed.
### The package does all the below steps as default 
   + Tokenize the multiple sentences into sentences using NLTK punkt tokenizer, if the input is a string of paragraph.
   + Expand Contractions like "y'all" to "you all".
   + remove Unkown Symbols including emoji, web_url patters etc.
   + Remove stopwords using NLTK stopwords, Spacy stopwords and few other combined.
   
### Extra parameters that can be given:
   + does POS tagging (pos=False/True)
   + Mask numbers as 'XnumberX' (mask_numbers=False/True)
   + perform lemmatization (lemma=False/True)
   + stop words removal (stop_words=True/False)
   + return word tokenized response (word_tokenize=False/True)
   + remove punctuations (remove_punct=True/False)
   
## Installation
   + Clone the repo under your root project directory.
   `git clone https://github.com/reaganrewop/text_preprocessing`
   + import the Preprocessor class.
   `from text_preprocessing import Preprocessor`
   
## Class `Preprocessor` contains entry function `get_preprocessed_text()`:
   
   + performs all the default functionalities that are mentioned above.
      
      `get_preprocessed_text(sentence :str or :list[str], lemma [default=False] :bool,  stop_words [default=False] :bool,  word_tokenize [default=False] :bool,  remove_punct [default=False] :bool,  pos [default=False] :bool, mask_numbers [default=False] :bool)`
      
      + Mandatory Args:
         + sentence: string of sentence(s) or list of sentence(s)
      + returns:
         + if pos = False
            + return list of sentence(s) 
         + if pos = True
            + list of sentence, where each sentence has list of tuple which contains (word, POS tag)
            

   ### Example
  
  ```
    from text_preprocessing import Preprocessor
    text = "Hello World!, This is a basic package on cleaning a text created at 6:03:2017."
    
    pt = Preprocessor()
    pt.get_preprocessed_text(text, mas_numbers=True)
  ```
  ### Output:
  
  >['Hello World!, This is a basic package on cleaning a text created at XnumberX. XnumberX. XnumberX']

