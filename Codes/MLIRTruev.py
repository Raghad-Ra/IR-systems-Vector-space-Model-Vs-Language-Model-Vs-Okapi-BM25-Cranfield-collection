import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import operator

class TermsInDoc:
   def __init__(self, docnum, dic):
            self.docnum = docnum
            self.dic = dic

def get_tokenized_list(doc_text):
    tokens = nltk.word_tokenize(doc_text)
    return tokens
def word_stemmer(token_list):
  ps = nltk.stem.PorterStemmer()
  stemmed = []
  for words in token_list:
    stemmed.append(ps.stem(words))
  return stemmed
stop_words = set(stopwords.words('english'))
# Function to remove stopwords from tokenized word list
def remove_stopwords(doc_text):
  cleaned_text = []
  for words in doc_text:
    if words not in stop_words and words.isalpha():
      cleaned_text.append(words)
  return cleaned_text

def tfincorpus(corpus):
    tfincorpus={}
    for document in corpus:
        for word in document.split():
            if word in tfincorpus:
                tfincorpus[word] = tfincorpus[word] + 1
            else:
                tfincorpus[word]=1
    return tfincorpus

def tfindocumentofcorpus(corpus):
    
    tfincorpusinall={}
    for document in corpus:
      tfincorpusindoc={}
      for word in document.split():
           if word not in tfincorpusindoc:
              tfincorpusindoc[word]=1
              if word in tfincorpusinall:
                tfincorpusinall[word]+=tfincorpusindoc[word]
              else:
                tfincorpusinall[word]=tfincorpusindoc[word] 
    return tfincorpusinall 

def tfindoc(corpus):
    tfindoclist=[]
    for document in corpus:
        tfindoc={}
        for word in document.split():
            if word in tfindoc:
                tfindoc[word]+=1
            else:
                tfindoc[word]=1
        tfindocItem =TermsInDoc(corpus.index(document),tfindoc)
        tfindoclist.append(tfindocItem)
    return tfindoclist

def tfindocRatio(corpus,tfindoc_list):
    tfindocRatioList=[]
    for document in corpus:
        tfindocRatio={}
        for word in document.split():
             for item in tfindoc_list:
                 if (item.docnum == corpus.index(document)):
                     tfindocRatio[word]=item.dic[word]/len(document.split())
        tfindocRatioItem =TermsInDoc(corpus.index(document),tfindocRatio)
        tfindocRatioList.append(tfindocRatioItem)
    return tfindocRatioList
def caluc_Pavg(tfindocRatio,df_d,allTokens):
  Pavg={}
  for token in allTokens:
      wordindoc=0
      for item in tfindocRatio:
        if token in item.dic :
          wordindoc+=item.dic[token]
      wordincorpus=df_d[token]
      Pavg[token]=wordindoc/wordincorpus
  return Pavg  

def termfreqmean (corpus,Pavg,tokens):
  termfreqmeanlist=[]
  for doc in corpus:
    for token in tokens:
      termfreqmean={}
      if token in Pavg :
        wordPavg=Pavg[token]
      termfreqmean[token]=wordPavg*len(doc.split()) 
      termfreqmeanItem=TermsInDoc(corpus.index(doc),termfreqmean)
      termfreqmeanlist.append(termfreqmeanItem) 
  return termfreqmeanlist

def riskofterm(termfreqmeanlist,corpus,tfindoc,allTokens):
  riskoftermlist=[]
  for doc in corpus:
    for word in allTokens:
      wordindoc=0
      riskofterm={}
      for doc_p in tfindoc:
        if word in doc_p.dic and doc_p.docnum==corpus.index(doc):
          wordindoc=doc_p.dic[word] 
      for item in termfreqmeanlist:
        if word in item.dic and  item.docnum==corpus.index(doc) :
          termfreqmean=item.dic[word]
          l1=(1/(1+termfreqmean))
          l2=(termfreqmean/(1+termfreqmean))**(wordindoc)
          riskofterm[word]=l1*l2
      riskoftermItem=TermsInDoc(corpus.index(doc),riskofterm)
      riskoftermlist.append(riskoftermItem)
  return riskoftermlist 

def termweight_ofcorpus(corpus,riskoftermlist,Pavg,tfindocRatioList,allTokens,Cs,tfincorpus):
  termweight_list=[]
  for document in corpus:
    for word in allTokens:
      wordindocRatio=0
      termweight={}
      for item in tfindocRatioList  : 
            if word in item.dic and item.docnum==corpus.index(document):
               wordindocRatio=item.dic[word]
      if wordindocRatio==0:
        wordincorpus=tfincorpus[word]
        termweight[word]=wordincorpus/Cs
      else:
        for item in riskoftermlist:
           if item.docnum==corpus.index(document) and word in item.dic : 
                 riskofterm=item.dic[word]  
                 l1=wordindocRatio**(1-riskofterm)
                 Pavrage=Pavg[word]
                 l2=Pavrage**(riskofterm)
                 termweight[word]=l1*l2
      termweightitem=TermsInDoc(corpus.index(document),termweight)
      termweight_list.append(termweightitem)
  return termweight_list

def termweight_ofquery(corpus,tfincorpus,query,Cs,termweight_ofcorpus_list,allTokens):
  documentProb_d={}
  for document in corpus:
    print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDd")
    queryindoc=[]
    L1=1
    L2=1
    for q in query:
        for item in termweight_ofcorpus_list:
          if item.docnum==corpus.index(document) and q in item.dic  : 
             L1*=item.dic[q]
             print(q)
             print(L1)
             print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
             queryindoc.append(q)
    for w in allTokens:
      if w not in queryindoc :
         for item in termweight_ofcorpus_list:
           if item.docnum==corpus.index(document) and w in item.dic : 
             L2*=(1-item.dic[w])   
             print(w)
             print(1-item.dic[w])
    print("L1")
    print(L1)
    print("L2")
    print(L2)
    documentProb_d[corpus.index(document)]=L1*L2 
  return documentProb_d

def LM(termweight_ofcorpus_d,termweight_ofquery_d,corpus):
  Document_p={}
  for document in corpus:
    L1=termweight_ofquery_d[corpus.index(document)]
    L2=1-termweight_ofcorpus_d[corpus.index(document)]
    Document_p[corpus.index(document)]=L1*L2     
  return Document_p

dataset = open("cran.all.1400.txt","r")#
text=dataset.read()
dataset.close()
name=re.split('.I \\d*\\n.T\\n',text)[1:] 
cleaned=[]
Cs=0
allTokens=[]
for doc in name:
     tokens = get_tokenized_list(doc)
     tokens = remove_stopwords(tokens)
     tokens  = word_stemmer(tokens)
     Cs=Cs+(len(tokens))
     for token in tokens:
         if token not in allTokens:
             allTokens.append(token)
     tokens = ' '.join(tokens) 
     cleaned.append(tokens)



tfindoc_list=tfindoc(cleaned)
tfindocRatio_list=tfindocRatio(cleaned,tfindoc_list)
tfincorpus_d=tfincorpus(cleaned)
df_d=tfindocumentofcorpus(cleaned)
print("1111111111111111111111111111111111111111111111111111111111111111111111")
caluc_Pavg_d=caluc_Pavg(tfindocRatio_list,df_d,allTokens)
print("2222222222222222222222222222222222222222222222222222222222222222222222")
termfreqmean_list=termfreqmean(cleaned,caluc_Pavg_d,allTokens)
print("3333333333333333333333333333333333333333333333333333333333333333333")
riskofterm_list=riskofterm(termfreqmean_list,cleaned,tfindoc_list,allTokens)
print("44444444444444444444444444444444444444444444444444444444444444444444")
termweight_ofcorpus_list=termweight_ofcorpus(cleaned,riskofterm_list,caluc_Pavg_d,tfindocRatio_list,allTokens,Cs,tfincorpus_d)
print("55555555555555555555555555555555555555555555555555555555555555555555")


#1 
q="""what similarity laws must be obeyed when constructing aeroelastic models
of heated high speed aircraft ."""
query = get_tokenized_list(q)
query = remove_stopwords(query)
query  = word_stemmer(query)

termweight_ofquery_d=termweight_ofquery(cleaned,tfincorpus_d,query,Cs,termweight_ofcorpus_list,allTokens)
sorted_d= dict(sorted(termweight_ofquery_d.items(), key=operator.itemgetter(1), reverse=True)[:10])
print(sorted_d)
#LM_d=LM(termweight_ofcorpus_list,termweight_ofquery_d,cleaned)
#print(LM_d)
#32
q="""can the three-dimensional problem of a transverse potential flow about
a body of revolution be reduced to a two-dimensional problem ."""
query = get_tokenized_list(q)
query = remove_stopwords(query)
query  = word_stemmer(query)

termweight_ofquery_d=termweight_ofquery(cleaned,tfincorpus_d,query,Cs,termweight_ofcorpus_list,allTokens)
sorted_d= dict(sorted(termweight_ofquery_d.items(), key=operator.itemgetter(1), reverse=True)[:10])
print(sorted_d)
#56
q="""what size of end plate can be safely used to simulate two-dimensional
flow conditions over a bluff cylindrical body of finite aspect ratio ."""
query = get_tokenized_list(q)
query = remove_stopwords(query)
query  = word_stemmer(query)

termweight_ofquery_d=termweight_ofquery(cleaned,tfincorpus_d,query,Cs,termweight_ofcorpus_list,allTokens)
sorted_d= dict(sorted(termweight_ofquery_d.items(), key=operator.itemgetter(1), reverse=True)[:10])
print(sorted_d)
#72
q="""what is a criterion that the transonic flow around an airfoil with a
round leading edge be validly analyzed by the linearized transonic flow
theory ."""
query = get_tokenized_list(q)
query = remove_stopwords(query)
query  = word_stemmer(query)

termweight_ofquery_d=termweight_ofquery(cleaned,tfincorpus_d,query,Cs,termweight_ofcorpus_list,allTokens)
sorted_d= dict(sorted(termweight_ofquery_d.items(), key=operator.itemgetter(1), reverse=True)[:10])
print(sorted_d)
#84
q="""can the three-point boundary-value problem for the blasius equation
be integrated numerically,  using suitable transformations,  without
iteration on the boundary conditions ."""
query = get_tokenized_list(q)
query = remove_stopwords(query)
query  = word_stemmer(query)

termweight_ofquery_d=termweight_ofquery(cleaned,tfincorpus_d,query,Cs,termweight_ofcorpus_list,allTokens)
sorted_d= dict(sorted(termweight_ofquery_d.items(), key=operator.itemgetter(1), reverse=True)[:10])
print(sorted_d)
