import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
#from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity

# this function returns a list of tokenized and stemmed words of any text
def get_tokenized_list(doc_text):
    tokens = set(nltk.word_tokenize(doc_text))
    return tokens

# This function will performing stemming on tokenized words
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

dataset = open("cran.all.1400.txt","r")#
text=dataset.read()
dataset.close()
name=re.split('.I \\d*\\n.T\\n',text)[1:] 
cleaned_corpus = []
for doc in name:
  tokens = get_tokenized_list(doc)
  doc_text = remove_stopwords(tokens)
  doc_text  = word_stemmer(doc_text)
  doc_text = ' '.join(doc_text)
  cleaned_corpus.append(doc_text)
cleaned_corpus
vectorizerX = TfidfVectorizer()
vectorizerX.fit(cleaned_corpus)
doc_vector = vectorizerX.transform(cleaned_corpus)
print(vectorizerX.get_feature_names())
print("=================================================================================================================")

df1 = pd.DataFrame(doc_vector.toarray(), columns=vectorizerX.get_feature_names())


#translator = Translator()
def retr_docs(query):
  #query = translator.translate(query, dest='en').text
  query = get_tokenized_list(query)
  query = remove_stopwords(query)
  q = []
  for w in word_stemmer(query):
    q.append(w)
  q = ' '.join(q)

  query_vector = vectorizerX.transform([q])
  cosineSimilarities = cosine_similarity(doc_vector,query_vector).flatten()
  related_docs_indices = cosineSimilarities.argsort()[:-11:-1]

  for i in range(10):
    print( related_docs_indices[i]+1001)
    print( cosineSimilarities[related_docs_indices[i]])



retr_docs('''is it possible to relate the available pressure distributions 
for an ogive forebody at zero angle of attack to the lower surface pressures
 of an equivalent ogive forebody at angle of attack .''')

