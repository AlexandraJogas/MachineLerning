import numpy as np
import math
import re

import nltk
from nltk.corpus import gutenberg,brown,swadesh  
from nltk.corpus import wordnet as wn

#1
#nltk.download()             
#nltk.download('gutenberg')  

#nltk.download('movie_reviews')
#nltk.download('stopwords')
from nltk.corpus import movie_reviews,stopwords
import random
import string

#2
file_name="austen-persuasion.txt"
raw =  gutenberg.raw(file_name)    
words =gutenberg.words(file_name)  
sents =gutenberg.sents(file_name)  
gutenberg.fileids()                


#3
top_sentence = """Today we will learn how to use the nltk (natural language toolkit) python library.
  This library enables us to perform numerous operations connected to natural language processing.  
  It contains corpora (databases of texts) and dictionaries in different langauges. 
  This library is the first and basic tool for nlp in python."""
from nltk.tokenize import sent_tokenize, word_tokenize
all_words = word_tokenize(top_sentence)  
trigrams = list(nltk.trigrams(all_words)) 
                                           
                                           
# 4
# laasot frequency distribution
words =gutenberg.words('bible-kjv.txt')   
fq    =nltk.FreqDist(words)    
res   = fq.freq('the')        
 
#5
common = fq.most_common(10)  

#6
words =brown.words()       
fq=nltk.FreqDist(words)    
words10 = [word for word, frequency in fq.items()  if frequency > 10] 

#7
from collections import defaultdict
print(swadesh.fileids())     
raw_sw=swadesh.raw('en')    
fr_letters={}               
fr_letters2=defaultdict(lambda: 1)
for i in range(len(raw_sw)):  
    l=raw_sw[i]
    if(l.isalpha()):    
        if l in fr_letters:
            fr_letters[l] += 1
            fr_letters2[l] +=1
        else:
            fr_letters[l] = 1
#assert fr_letters==dict(fr_letters2)
         
#8
h=wn.synsets('motorcar')  
all_synsets = wn.all_synsets('n') 
nb_syn = sum(1 for s in wn.all_synsets('n'))   # sah akol
fac = len([s for s in wn.all_synsets('n') if len(s.hyponyms()) == 0]) / nb_syn  
                                                                                
#9      
total = i = 0
for s in wn.all_synsets('n'):  
    ln = len(s.hyponyms())      
    if ln != 0 :
      total += ln
      i += 1
percent = total / i           
        
    
#10
par="""In computer science, lexical analysis, lexing or tokenization is the process of converting
     a sequence of characters (such as in a computer program or web page) into a sequence of tokens
     (strings with an assigned and thus identified meaning). A program that performs lexical analysis
     may be termed a lexer, tokenizer,[1] or scanner, though scanner is also a term for the first
     stage of a lexer. A lexer is generally combined with a parser, which together analyze the syntax
     of programming languages, web pages, and so forth."""
     
from nltk.tokenize import sent_tokenize, word_tokenize   
#nltk.download('punkt')
stemmed_words = []
porter        = nltk.PorterStemmer()   
sent_tok_list = sent_tokenize(par)
for sentence in sent_tok_list:                   
    word_tok_list =word_tokenize(sentence)
    for word in word_tok_list:                    
        stemmed_words.append(porter.stem(word))  
        
#11
from nltk import pos_tag           
all_words = word_tokenize(par)      
word_tags = pos_tag(all_words)     

#12
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer  = WordNetLemmatizer()  
lematized = []                             
for word in all_words:
    lematized.append(wordnet_lemmatizer.lemmatize(word))  


#13
# tf-idf
    # likro kvazim ve leyazer data
categories = movie_reviews.categories()  
documents = [(list(movie_reviews.words(fileid)), category)  
                   for category in categories              
                   for fileid in movie_reviews.fileids(category)] 

random.shuffle(documents) 
#all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
all_words = nltk.FreqDist(                          
   wordnet_lemmatizer.lemmatize(w.lower()) for w in movie_reviews.words()  
   if w.lower() not in stopwords.words('english') and  
      w.lower() not in list(string.punctuation)   and w.isalpha())  

        
#TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
#TF * IDF = TF * IDF

bow_dict = {} 
words_doc=[]   
docs=np.asarray(documents)[:,0] 
total_doc =len(docs)  
for i in range(total_doc):  
    words_doc.append(nltk.FreqDist(  
        wordnet_lemmatizer.lemmatize(w.lower()) for w in docs[i]  
            w.lower() not in list(string.punctuation)   and w.isalpha())) 
    total_words_doc=sum(words_doc[i][key]  for key in words_doc[i]) 
#    print(total_words_doc)
    for key in words_doc[i]:
        #print (key)       
        words_doc[i][key]=words_doc[i][key]/total_words_doc #TF(t)= (Number of times term t appears in a document) / (Total number of terms in the document)
        if key in bow_dict:
            bow_dict[key] += 1    
            else:
            bow_dict[key] = 1
            
for i in range(total_doc):
    for key in words_doc[i]: 
        words_doc[i][key]=words_doc[i][key]*math.log(len(docs)/bow_dict[key])  # TF*IDF= TF(t)*log(2000/50)
        
#14   
word_features = list(all_words)[:1000]   
def document_features(document): 
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = word in document_words  
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]  
random.shuffle(featuresets)
train_set, test_set = featuresets[100:], featuresets[:100]  
classifier = nltk.NaiveBayesClassifier.train(train_set)    

#15
accuracy = nltk.classify.accuracy(classifier, test_set)

#16
pos_review = '''Amazing. I loved it. what a movie. want to watch it again'''
neg_review = '''The movie didn't feel real to me. it was fake'''

wordListPos = re.sub("[^\w]", " ",  pos_review).split()
wordListNeg = re.sub("[^\w]", " ",  neg_review).split()
mydocuments=[(wordListPos,'pos'),(wordListNeg,'neg')]
#my_all_words = nltk.FreqDist(
#        wordnet_lemmatizer.lemmatize(w.lower()) for w in wordListPos +wordListNeg
#        if w.lower() not in stopwords.words('english') and
#            w.lower() not in list(string.punctuation) and w.isalpha())

#word_features = list(my_all_words)[:2]
my_test_set= [(document_features(d), c) for (d,c) in mydocuments]
print(nltk.classify.accuracy(classifier, my_test_set))
