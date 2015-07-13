#I started by using code from the following authors

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sys import exit

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

###############################################################################
# Load all categories from the training set
categories = None

#remove  headers, signatures, and quoting to avoid overfitting
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

print('data loaded')

categories = data_train.target_names    # for case categories == None

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(categories))
print()

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()

#if you want to build custom tokenziers
import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer(object):
  def __init__(self):
    self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
      return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#for some reason PorterStemmer is too slow
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = TfidfVectorizer( #tokenizer=tokenize, #provide a tokenizer if you want to
                              sublinear_tf=True, 
                              use_idf=True,
                              max_df=0.5,      
                              min_df = 10.0/11314.0, #words must appear at least 10 times, 11314 is the number of total words 
                              stop_words='english')

X_train = vectorizer.fit_transform(data_train.data)

duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

print(type(X_test))
print(type(y_test))

#export as a sparse matrix for use in external software
def export_matrix(file_name, csr_matrix, labels):
  (nrows, ncols) = csr_matrix.shape
  import codecs

  f = codecs.open(file_name, "w", "utf-8")

  for i in range(0, nrows):
    indices = csr_matrix.indices[csr_matrix.indptr[i]:csr_matrix.indptr[i+1]] 
    values = csr_matrix.data[csr_matrix.indptr[i]:csr_matrix.indptr[i+1]]
    f.write("label:" + str(labels[i]))
    f.write("\t" + "caseId:" + str(i))    
    for j in range(0, len(indices)):
      index = indices[j]
      feature_name = feature_names[index]      
      value = values[j]
      f.write("\t" + feature_name + ":" + str(value))
    f.write("\n")
  f.close()
   
#export_matrix("/tmp/20newgroups_train.csv", X_train, y_train)
#export_matrix("/tmp/20newgroups_test.csv", X_test, y_test)

import numpy as np
from sklearn.decomposition import ProjectedGradientNMF

#compute the SVD using a sparse matrix
import scipy
import numpy
svd_output = scipy.sparse.linalg.svds(X_train, k=200, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True)
U,d,Vt = svd_output
print('Vt' + str(type(Vt)))
print(Vt.shape)

#warning: the most significant singular values/vectors are at k = 199

compute_example_nearest_neighbors = True

if compute_example_nearest_neighbors:
    s = d
    V = Vt
    word_vectors = feature_names
    Sinv = np.diag(s/(s + 1))
    word_vectors = V.T.dot(Sinv)
    norms = []
    num_words = word_vectors.shape[0]
    for i in range(0, num_words):
        v = word_vectors[i, :]
        norms.append(np.sqrt(v.dot(v.T)))
            
    def word_nearest_neighbors(word):
        word_id = -1
        i = 0
        
        while (i < len(feature_names) and word_id == -1):
          if (feature_names[i] == word):
            word_id = i
          i += 1

        query = word_vectors[word_id, :]
        word_scores = word_vectors.dot(query.T)
        
        id_and_scores = []
        i = 0
        for i in range(0, len(word_scores)):
            id_and_scores.append((i, word_scores[i]))
        
        results = sorted(id_and_scores, key = lambda (k,v): -v)        
        print("-----------------------------")
        print("neighbors of word " + word)
        for i in range(0, 10):
            (k,v) = results[i]
            print(str(i) + ")" + feature_names[k] + "\t" + str(v))  
            
    word_nearest_neighbors('windows')
    word_nearest_neighbors('jesus')
    word_nearest_neighbors('congress')
    

#save the matrices U,V,S for plotting in R
np.savetxt('U.txt', U, delimiter='\t')

np.savetxt('Vt.txt', Vt, delimiter='\t')

np.savetxt('S.txt', d, delimiter='\t')

#dump some "patterns" computed by the SVD
for r in range(190, 200):
  row = Vt[r, :]

  l_row = list(row)
  print('\nrow - ' + str(r))
  lst = []
  for i in range(0, len(l_row)):
    value = l_row[i]
    lst.append((feature_names[i], value))

  for(k,v) in sorted(lst, key=lambda (k,v): v)[0:10]:
    print(k + "\t" + str(v))

  print('\nrow + ' + str(r))

  for(k,v) in sorted(lst, key=lambda (k,v): -v)[0:10]:
    print(k + "\t" + str(v))


from scipy.sparse import csr_matrix
print(X_train.shape)
print(csr_matrix(Vt.T).shape)
diag_m = csr_matrix(numpy.diag(np.sqrt(1.0/(d + 1.0))))

sys.exit(1)

#proceed if you want to compute classification with/without SVD
X_train_svd = X_train.dot(csr_matrix(Vt.T)) 
X_test_svd = X_test.dot(csr_matrix(Vt.T)) 

classfication_with_svd = False

if classfication_with_svd:
  X_train = X_train_svd
  X_test = X_test_svd

import matplotlib.pyplot as plt
import numpy as np
#example with histogram plotting
#hist, bins = np.histogram(row, bins=50)
#width = 0.9 * (bins[1] - bins[0])
#center = (bins[:-1] + bins[1:]) / 2
#plt.bar(center, hist, align='center', width=width)
#plt.show()



#TO DO: for some reason this code did not work well. Find out why
#from sklearn.decomposition import TruncatedSVD
#print("SVD")
#svd = TruncatedSVD(algorithm='randomized', n_components=100, random_state=42)
#svd.fit(X_train) 
#X_train = svd.fit_transform(X_train)
#X_test = svd.fit_transform(X_test)
#print(type(svd.get_params()))
#print(svd.get_params().keys())
#print(svd.explained_variance_ratio_.sum())


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()

    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)

    
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        print()

    print("classification report:")
    print(metrics.classification_report(y_test, pred,
                                        target_names=categories))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive")): #,
        #(KNeighborsClassifier(n_neighbors=10), "kNN")): #,
        #(RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
#print("Naive Bayes")
#results.append(benchmark(MultinomialNB(alpha=.01)))
#results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
