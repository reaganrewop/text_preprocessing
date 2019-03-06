# text_preprocessing

### The package does all the below steps as default 
   + Tokenize the multiple sentences into sentences using NLTK punkt tokenizer.
   + Expand Contractions like "y'all" to "you all".
   + remove Unkown Symbols and punctuations.
   + replace numbers which exist alone as 'XnumberX'
   + Remove stopwords using NLTK and Spacy list combined.
   
#### INPUT:  string sentence(s).
#### OUTPUT:  list of sentence(s). 

## Walkthrough

   + Clone the repo
   `git clone https://github.com/reaganrewop/text_preprocessing`
   
   + import the package in your code file and call the preprocess function passing a string of sentence.
  
  ```
    from text_preprocessing import preprocess
    preprocess.preprocess("Hello World!. This is a basic package on cleaning a text created at 6:03:2017 7:54.")
  ```
  ### Output:
  >[' Hello World ', ' This basic package cleaning text created XnumberX XnumberX ']

