import os
import math
import json
import requests
import re
import operator
dataset = open("cran.all.1400.txt","r")#
text=dataset.read()
dataset.close()
name=re.split('.I \\d*\\n.T\\n',text)[1:] 
corpus=[]
for doc in name:
     for token in name:
         corpus.append(token)

# remove stop words and tokenize them (we probably want to do some more
# preprocessing with our text in a real world setting, but we'll keep
# it simple here)
stopwords = set(['for', 'a', 'of', 'the', 'and', 'to', 'in'])
texts = [
    [word for word in document.lower().split() if word not in stopwords]
    for document in corpus
]

# build a word count dictionary so we can remove words that appear only once
word_count_dict = {}
for text in texts:
    for token in text:
        word_count = word_count_dict.get(token, 0) + 1
        word_count_dict[token] = word_count

texts = [[token for token in text if word_count_dict[token] > 1] for text in texts]
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            tf.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query):
        scores = [self._score(query, index) for index in range(self.corpus_size_)]
        return scores

    def _score(self, query, index):
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)

        return score
# query our corpus to see which document is more relevant
query = '''what similarity laws must be obeyed when constructing aeroelastic models
of heated high speed aircraft .'''
query = [word for word in query.lower().split() if word not in stopwords]

bm25 = BM25()
bm25.fit(texts)
scores = bm25.search(query)
d={}
for score, doc in zip(scores, corpus):
    score = round(score, 3)
    d[corpus.index(doc)]=score
sorted_d= dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:50])
print(sorted_d)










