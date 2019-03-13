# Text_Preprocessing
## A light-weight text preprocessing package which does what is most needed.
### The package does all the below steps as default 
   + Tokenize the multiple sentences into sentences using NLTK punkt tokenizer.
   + Expand Contractions like "y'all" to "you all".
   + remove Unkown Symbols and punctuations.
   + replace numbers which exist alone as 'XnumberX'
   + Remove stopwords using NLTK and Spacy list combined.
   
### Extra parameters that can be given:
   + perform lemmatization (lemma=False/True)
   + stop words removal (stop_words=True/False)
   + return word tokenized response (word_tokenize=False/True)
   + remove punctuations (remove_punct=True/False)
   + do POS tagging (pos=False/True)
   
## Installation
   + Clone the repo under your root project directory.
   `git clone https://github.com/reaganrewop/text_preprocessing`
   + import the package and use for your project.
   `from text_preprocessing import preprocess`
   
## function `preprocess`:
   
   + performs all the default functionalities that are mentioned above.
      
      `preprocess(sentence :str, lemma [default=False] :bool,  stop_words [default=True] :bool,  word_tokenize [default=False] :bool,  remove_punct [default=True] :bool,  pos [default=False] :bool)`
      
      + Args:
         + sentence: string of sentence(s)
      + returns:
         + if pos = False
            + return list of sentence(s) 
         + if pos = True
            + list of sentence, where each sentence has list of tuple which contains (word, POS tag)
            

   ### Example
  
  ```
    import text_preprocessing.preprocess
    text_preprocessing.preprocess.preprocess("Hello World!. This is a basic package on cleaning a text created at 6:03:2017 7:54.")
  ```
  ### Output:
  
  >[' Hello World ', ' This basic package cleaning text created XnumberX XnumberX ']

## function `get_filtered_pos`:

   + filter words based on the given pos. if word pos is not present on the given list then remove.
   
     `get_filtered_pos(sentences :list[:list[:tuple(:str,:str)]], filter_pos=['ADJ', 'VERB', 'NOUN', 'PROPN', 'FW'] :list[:str])`
     
      + Args: 
         + sentence: list of sentences, where each sentence consist of a tuple of (word, pos)
         + filter_pos: list of POS tags that needs to be grepped from the original sentence.
      + returns:
         + A list of sentences, where each sentence consist of a tuple of filtered (word,pos)
         
  ### Example
  
  ```
    import text_preprocessing.preprocess
  
    text = "Hello World!. This is a basic package on cleaning a text created at 6:03:2017 7:54."
    
    tokenized_Sent, word_pos_list = text_preprocessing.preprocess.preprocess(text, stop_words=True, remove_punct=False, word_tokenize=True, pos = True)
    
    filtered_pos = text_preprocessing.preprocess.get_filtered_pos(word_pos_list, filter_pos=['ADJ', 'VERB', 'NOUN', 'PROPN', 'FW'])
    
    print (filtered_pos)
  ```
  ### Output:
  
  >[
  
   >[ ('World', 'NOUN') ],
  
   >[ ('basic', 'ADJ'), ('package', 'NOUN'), ('cleaning', 'VERB'), ('text', 'NOUN'), ('created', 'VERB') ]
  
  >]

