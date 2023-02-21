# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:25:30 2018

@author: elena
"""
import numpy as np
import math
import re

import nltk
from nltk.corpus import gutenberg,brown,swadesh  # library database textim
from nltk.corpus import wordnet as wn

#1
#nltk.download()             dereh rishona lehorid
#nltk.download('gutenberg')  dersh yashira morid le mhshev

#nltk.download('movie_reviews')
#nltk.download('stopwords')
from nltk.corpus import movie_reviews,stopwords
import random
import string

#2
#pkudot she maviet data be tazuga shona: text, milim, mishpatim
file_name="austen-persuasion.txt"
raw =  gutenberg.raw(file_name)    # likro kovez mesuyam me rashimat kvazim, roim text le mehulak
words =gutenberg.words(file_name)  # lekabel milim me file haze
sents =gutenberg.sents(file_name)  # likro mishpatim
gutenberg.fileids()                # mavi rashima shel kol kvazim she nimzaim be gutenberg 


#3
top_sentence = """Today we will learn how to use the nltk (natural language toolkit) python library.
  This library enables us to perform numerous operations connected to natural language processing.  
  It contains corpora (databases of texts) and dictionaries in different langauges. 
  This library is the first and basic tool for nlp in python."""
from nltk.tokenize import sent_tokenize, word_tokenize
all_words = word_tokenize(top_sentence)  
trigrams = list(nltk.trigrams(all_words))  # lakahat text ve lehozia shlishiiat milim ze feature=trigram
                                           # esh Ngram= n milim
                                           
# 4
# laasot frequency distribution
words =gutenberg.words('bible-kjv.txt')    # lakahnu me gutenberg('bible) ve yazagnu be ezrat milim
fq    =nltk.FreqDist(words)    # kama paamim mofia kol mila mi toh text 
res   = fq.freq('the')         # kama paamim mila "the" mofia, ma freq yahasi shel mila "the" =0.0614
 
#5
common = fq.most_common(10)  # limzo 10 milim ahi nafuzim

#6
words =brown.words()       # mekablim kol milim me text brown she nimza be gutenberg
fq=nltk.FreqDist(words)    # osim frequesty distribution kama paamim kol mila mofia 
words10 = [word for word, frequency in fq.items()  if frequency > 10] # mahzir kol milim she freq gadol mi 10

#7
from collections import defaultdict
print(swadesh.fileids())     # swadesh mehil kama kvazim ve kama safot, nikah rak english
raw_sw=swadesh.raw('en')     # english= safa
fr_letters={}                # bonim dictionary: im ot mofia sofrim +1, im le roshmim ota
fr_letters2=defaultdict(lambda: 1)
for i in range(len(raw_sw)):   # laavor al kol string aroh
    l=raw_sw[i]
    if(l.isalpha()):    # bdika im alfa ki meanien otanu milim ve le misparim
        if l in fr_letters:
            fr_letters[l] += 1
            fr_letters2[l] +=1
        else:
            fr_letters[l] = 1
#assert fr_letters==dict(fr_letters2)
         
#8
h=wn.synsets('motorcar')  # laavor al kol sinonims ve livdok kama ahuz mi hem hyponyms, milim kmo ambulance le rehev
all_synsets = wn.all_synsets('n')  # mavia kol sinonim le motorcar nouns
nb_syn = sum(1 for s in wn.all_synsets('n'))   # sah akol
fac = len([s for s in wn.all_synsets('n') if len(s.hyponyms()) == 0]) / nb_syn  # hyponyms= lavaor mi klali le prati
                                                                                # kama ahuz sinonim ein lahem hyponyms
#9      
total = i = 0
for s in wn.all_synsets('n'):   # synsets= milim im ota mashmaut sinonim
    ln = len(s.hyponyms())      # hyponyms= erarhia beim milim
    if ln != 0 :
      total += ln
      i += 1
percent = total / i            # memuz shel hyponyms
        
    
#10
par="""In computer science, lexical analysis, lexing or tokenization is the process of converting
     a sequence of characters (such as in a computer program or web page) into a sequence of tokens
     (strings with an assigned and thus identified meaning). A program that performs lexical analysis
     may be termed a lexer, tokenizer,[1] or scanner, though scanner is also a term for the first
     stage of a lexer. A lexer is generally combined with a parser, which together analyze the syntax
     of programming languages, web pages, and so forth."""
     
from nltk.tokenize import sent_tokenize, word_tokenize   # lehalek le mishpatim ve milim txt
#nltk.download('punkt')
stemmed_words = []
porter        = nltk.PorterStemmer()   # leagdir PorterStemmer kdei lahtoh milim kom going=go, stars=star
sent_tok_list = sent_tokenize(par)
for sentence in sent_tok_list:                    # lehalek kol mishpatim le milim
    word_tok_list =word_tokenize(sentence)
    for word in word_tok_list:                    # al kol milim naase stemmed_words= lahtov lesader mila mi yamin ve smol
        stemmed_words.append(porter.stem(word))   # going= go, neigbours=neigbor
        
#11
from nltk import pos_tag            # part of speech=POS= be eize helek shel mishpat anu nimzaim, noun, adjective
all_words = word_tokenize(par)      # lefarek le list shel milim, mishpatim
word_tags = pos_tag(all_words)      # (is : JJ), (comp: NN)

#12
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer  = WordNetLemmatizer()  # going=go, supering=super
lematized = []                             # list hadash meyazrim
for word in all_words:
    lematized.append(wordnet_lemmatizer.lemmatize(word))   # wordnet_lemmatizer= lehazig kol shorashim shel milim


#13
# tf-idf
    # likro kvazim ve leyazer data
categories = movie_reviews.categories()   # movie_reviews= database muhan, haim seret tov or le tov, nesader data kah: 
documents = [(list(movie_reviews.words(fileid)), category)  # tazig list(milim,neg/pos) = ze data shelanu
                   for category in categories               # lokhim rak categories: negative and positive 
                   for fileid in movie_reviews.fileids(category)] # ovrim al rashimat kvazim= ve moziim list milim

random.shuffle(documents)  # nearbev positive and negative
#all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
all_words = nltk.FreqDist(                             # frequesty distribution kama paamim kol mila mofia 
   wordnet_lemmatizer.lemmatize(w.lower()) for w in movie_reviews.words()  # wordnet_lemmatizer= lehazig kol shorashim shel milim
   if w.lower() not in stopwords.words('english') and  # im milim le bestopwords, she lerozim lehishtamesh bahem the, is, a...
      w.lower() not in list(string.punctuation)   and w.isalpha())  # bdika she ze not punctuation, ve she ze mila= isalpha

        
#TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
#TF * IDF = TF * IDF

bow_dict = {}  # lishmor lekol mismah: haim mila mofia bo= (Number of documents with term t in it)
words_doc=[]   # kama paamim mila mofia be mismah
docs=np.asarray(documents)[:,0]  # meamirim le array ve lokhim amuda rishona, le lokhim pos, negative
total_doc =len(docs)    # lenght=2000
for i in range(total_doc):  # laavor be kol documents= 2000
    words_doc.append(nltk.FreqDist(   # frequesty distribution kama paamim kol mila mofia
        wordnet_lemmatizer.lemmatize(w.lower()) for w in docs[i]  # lemmatize=lokhim shoresh
        if  w.lower() not in stopwords.words('english') and       # bdika im milim le be stopwords= is, the, 
            w.lower() not in list(string.punctuation)   and w.isalpha())) # bdika she ze not punctuation, ve she ze mila= isalpha
    total_words_doc=sum(words_doc[i][key]  for key in words_doc[i]) # godel mismah=leshom kol milim be i=mismah: words_doc[i][key]=eize mismah, eize mila
#    print(total_words_doc)
    for key in words_doc[i]:
        #print (key)        # leadken words_doc[i][key]: bimkom 5, ihie 5/350
        words_doc[i][key]=words_doc[i][key]/total_words_doc #TF(t)= (Number of times term t appears in a document) / (Total number of terms in the document)
        if key in bow_dict:
            bow_dict[key] += 1   # sofrim mispar mismahim she mila mofia = Number of documents with term t in it (for IDF)
            else:
            bow_dict[key] = 1
            
for i in range(total_doc):
    for key in words_doc[i]: 
        words_doc[i][key]=words_doc[i][key]*math.log(len(docs)/bow_dict[key])  # TF*IDF= TF(t)*log(2000/50)
        
#14   
# naive bayes classifaers from nltk, hu yodea lekabel milim, ve lesader features, ve leariz classifier
word_features = list(all_words)[:1000]   
def document_features(document): 
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = word in document_words   # mahzira {true/false} im mila mofia be document= mismah
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]  # baninu documents(milim:true/false, negative/positive)
random.shuffle(featuresets)
train_set, test_set = featuresets[100:], featuresets[:100]  # lehalek data test, train
classifier = nltk.NaiveBayesClassifier.train(train_set)     # laasot Classifier

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