from sklearn.datasets import fetch_mldata
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib
import pylab as P

#DO NOT USE: highly experimental stuff

def show_vector_plot(flattened_image):
    image = np.reshape(flattened_image, (-1, 28))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show() 

def plot_vector_png(fname, flattened_image):
  fig = matplotlib.pyplot.gcf()
  fig.set_size_inches(0.5, 0.5)
  image = np.reshape(flattened_image, (-1, 28))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(image, cmap = matplotlib.cm.binary)
  plt.xticks(np.array([]))
  plt.yticks(np.array([]))
  filename = fname + ".png" 
  fig.savefig(filename, dpi=200)

def load_images():
    mat = np.ndarray(shape=(42000, 784), dtype=float, order='F')  
    f = open('/home/stefan2/kaggle/lsh/E2LSH-0.1/normalized.txt', 'r')
    data = []
    i = 0    
    for line in f.xreadlines():
      values = map(float, line.split(' '))
      #print(len(values))      
      for j in range(0, 784):
        mat[i, j] = values[j]       
      i = i + 1    
    f.close()
    return mat
#    np.ndarray(shape=(1, m), dtype=float, order='F')[0] 

def show_vector_plot(flattened_image):
    image = np.reshape(flattened_image, (-1, 28))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def plot_vector_png(filename, flattened_image):
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(0.5, 0.5)

    image = np.reshape(flattened_image, (-1, 28))

    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

    filename = "fig_" + filename + ".png" 
    fig.savefig(filename, dpi=150)

def load_labels():
  f = open('/home/stefan2/kaggle/lsh/E2LSH-0.1/labels.txt', 'r')
  labels = []    
  for line in f.xreadlines():
    labels.append(int(line))
  f.close()
  return labels

from scipy.sparse import csr_matrix
import numpy as np
import scipy
import scipy.sparse.linalg as la

def load_neighbors(labels):
    rows = [] #document ids
    cols = [] #word ids
    scores = [] #score for (doc_id, word_id)
    s = set()
    f = open('/home/stefan2/kaggle/lsh/E2LSH-0.1/neighbors.txt', 'r')
    for line in f.xreadlines():
      parts = line.split(' ')
      a = int(parts[0])
      b = int(parts[1])
      value = 1.0 - float(parts[2]) #now the score is euclidean dist
      #a huge direction obscures the rest
      #if labels[a] <> 1 and labels[b] <> 1 and labels[a] <> 0 and labels[b] <> 0:      
      s.add(a)      
      rows.append(a)
      cols.append(b)
      scores.append(value)
      
      rows.append(b)
      cols.append(a)
      scores.append(value)      
    
    f.close()
    
    n = 42000 #max(list(s))
    print("number of dimensions: " + str(n))
    data = csr_matrix((scores, (rows, cols)), shape=(n, n))
    return ((rows, cols, scores), data)

def filter_data(rows,cols, scores, selected):
  new_rows = []
  new_cols = []
  new_scores = []

  for i in range(0, len(rows)):
    a = rows[i]
    b = cols[i]
    value = scores[i]
    if a in selected and b in selected:
      new_rows.append(a)
      new_cols.append(b)
      new_scores.append(value)
  n = 42000 #max(list(s))
  data = csr_matrix((new_scores, (new_rows, new_cols)), shape=(n, n))
  return data
          
import matplotlib.pyplot as plt

from pylab import *

def initialize_cluster_centers(input_data, labels, images):
  ((rows, cols, input_scores), neighbors) = input_data
  n,m = neighbors.shape
  #point_ids = range(0, n)
  selected_ids = set(range(0, n))
  i = 0
  #for i in range(0, 20):
  while (len(selected_ids) > 5000):
    svd_result = la.svds(neighbors, k=50, ncv=None, tol=0, which='LM', v0=None, maxiter=None,return_singular_vectors=True)
    U, s, V = svd_result

    scores = list(U[:,U.shape[1] - 1]) #this is the wrong 0 should be num_cols - 1
    ids = range(0, len(scores))
    print '---------------------------------'
    print ('cluster_' + str(i))
    sorted_values_ids = sorted(zip(ids,scores), key=lambda (_id,value): value)
    for (_id,v) in sorted_values_ids[1:10]:
      if abs(v) > 0.05:
        lab = labels[_id]
        print("%f  %d %d" % (v, _id, lab))
        plot_vector_png("cluster_" + str(i) + "_" + str(_id), images[_id,:])
      else:
        print("skipping too small abs. value")
    sorted_values_ids = sorted(zip(ids,scores), key=lambda (_id,value): -value)

    for (_id,v) in sorted_values_ids[1:10]:
      if abs(v) > 0.05:
        lab = labels[_id]
        print("%f  %d %d" % (v, _id, lab))
        plot_vector_png("neg_cluster_" + str(i) + "_" + str(_id), images[_id,:])
      else:
        print("skipping too small abs. value")

    for (_id,v) in sorted_values_ids[1:2500]:
      if (_id in selected_ids):
        selected_ids.remove(_id)
    i = i + 1
    neighbors = filter_data(rows, cols, input_scores, selected_ids)
    
def read_vocabulary():
  word_ids_file = "/home/stefan2/kaggle/wordvecs/code/trunk/word_ids"
  f = open(word_ids_file , 'r')
  word2_id = {}   
  for line in f.xreadlines():
    parts = line.replace("query: ", "").split(" ")
    word_id = int(parts[1])
    if True: #(word_id < 20000):
      word2_id[parts[0]] = word_id
  f.close()
  return word2_id

def read_neighbors():
  word2_id = read_vocabulary()

  rows = []
  cols = []
  scores = []
  
  neighbors_file = "/home/stefan2/kaggle/wordvecs/code/trunk/neighbors"

  f = open(neighbors_file, 'r')
  for line in f.xreadlines():
    parts = line.replace("nn: ", "").split(" ")
    word_a = parts[0]
    word_b = parts[2]
    if (word2_id.has_key(word_a) and word2_id.has_key(word_b)):
      a = word2_id[parts[0]]
      b = word2_id[parts[2]]
      
      value = float(parts[3]) #now the score is euclidean dist
      value = value*value*value*value
      if (a == b):
        value += 0.1 #1.0
      
      rows.append(a)
      cols.append(b)
      scores.append(value)

      if (a != b):
        rows.append(b)
        cols.append(a)
        scores.append(value)      
    
  f.close()
    
  n = len(word2_id)
  print("number of dimensions: " + str(n))
  data = csr_matrix((scores, (rows, cols)), shape=(n, n))
  return ((rows, cols, scores), data, word2_id)

def compute_nearest_neighbors(U, d, feature_names, word):
    s = d
    #for i in range(0, len(s)):
    #  print("s(" + str(i) + ") " + str(s[i]))
    #V = Ut
    #word_vectors = feature_names
    Sinv = np.diag(s/(s + 1))
    word_vectors = U.dot(Sinv) #.dot(np.diag(s)) #
    #norms = []
    num_words = word_vectors.shape[0]
    #for i in range(0, num_words):
    #    v = word_vectors[i, :]
    #    norms.append(np.sqrt(v.dot(v.T)))
            

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
            
    word_nearest_neighbors(word)
    #word_nearest_neighbors('jesus')
    #word_nearest_neighbors('congress')

def compute_nearest_neighbors2(word_vectors, feature_names, word):
    print("converting to dense")

    #norms = []
    num_words = word_vectors.shape[0]
    #for i in range(0, num_words):
    #    v = word_vectors[i, :]
    #    norms.append(np.sqrt(v.dot(v.T)))
            

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
        for i in range(0, 50):
            (k,v) = results[i]
            print(str(i) + ")" + feature_names[k] + "\t" + str(v))  
            
    word_nearest_neighbors(word)
    #word_nearest_neighbors('jesus')
    #word_nearest_neighbors('congress')

def compute_nearest_neighbors3(query_vectors, word_vectors, feature_names, word):
    #print("converting to dense")

    #norms = []
    num_words = word_vectors.shape[0]
    #for i in range(0, num_words):
    #    v = word_vectors[i, :]
    #    norms.append(np.sqrt(v.dot(v.T)))
            

    def word_nearest_neighbors(word):
        word_id = -1
        i = 0
        
        while (i < len(feature_names) and word_id == -1):
          if (feature_names[i] == word):
            word_id = i
          i += 1

        query = query_vectors[word_id, :]
        word_scores = word_vectors.dot(query.T)
        
        id_and_scores = []
        i = 0
        for i in range(0, len(word_scores)):
            id_and_scores.append((i, word_scores[i, 0]))
            #id_and_scores.append((i, word_scores[i]))

        results = sorted(id_and_scores, key = lambda (k,v): -v)        
        print("-----------------------------")
        print("neighbors of word " + word)
        for i in range(0, 50):
            (k,v) = results[i]
            print(str(i) + ")" + feature_names[k] + "\t" + str(v))  
            
    word_nearest_neighbors(word)
    #word_nearest_neighbors('jesus')
    #word_nearest_neighbors('congress')

import itertools

def do_word_vectors(script_vars):
  def define(var_name, fun, overwrite=False):
    if script_vars.has_key(var_name) and not overwrite:
      print('%s is already defined' % var_name)
      return script_vars[var_name]
    else:
      print('computing variables %s' % var_name)
      value = fun()
      script_vars[var_name] = value
      globals()[var_name] = value      
      return value

  print('svd on thresholded word vectors')
  word_vector_data = define('word_vector_data', lambda: read_neighbors(), False)
  ((rows, cols, scores), data, word2_id) = word_vector_data

  labels = [None]*len(word2_id)
  for (k,v) in word2_id.iteritems():
    labels[v] = k
 
  #do summary
  #loop over rows of matrix, choose a partition (row in summary) choose a sign, add the vector
  #need to change it: given a row index predetermine row in summary and sign
  #on (i,j,v): i => [row, sign] => summary[row, j] += sign*v
  
  summary_entries = {}

  #if there a lasso method with a very large number of predictors, if so this would be easy
  num_entries_summary = 1000
  row_mapping = [None]*len(word2_id)
  summary_signs = [None]*len(word2_id)
  for (i, perm_id) in zip(range(0, len(word2_id)), np.random.permutation(len(word2_id))):
    row_mapping[i] = (perm_id - 1) % num_entries_summary
    summary_signs[i] = 2.0*(np.random.choice(2,1)[0] - 0.5) #+/-1 with 0.5

  #cx = x.tocoo()    
  for i,j,v in itertools.izip(rows, cols, scores): #data.row, data.col, data.data):
    row=row_mapping[i]
    sign = summary_signs[i]
    k = (row,j)    

    if summary_entries.has_key(k):
      summary_entries[k] += sign*v
    else:
      summary_entries[k] = sign*v    
      
  print("done with summary matrix")
  summary_rows = []
  summary_cols = []
  summary_values = []

  for ((i,j),v) in summary_entries.iteritems():
    summary_rows.append(j) #feature id 
    summary_cols.append(i) #summary row
    summary_values.append(v)
        
  summary_matrix = csr_matrix((summary_values, (summary_rows, summary_cols)), shape=(len(word2_id), num_entries_summary))
  
  sparsity = 100.0*len(summary_values)/(len(word2_id) * num_entries_summary)
  print("Sparsity: " + str(sparsity))
  print("done with summary matrix")
  proj_data = data.dot(summary_matrix)


  #summary_matrix is (n by k)
  d = [1.0]*num_entries_summary
  to_inv = summary_matrix.T.dot(summary_matrix) + np.diag(d) #has the effect to normalize the scores
  query_repr = summary_matrix.dot(np.linalg.inv(to_inv)) #.toarray() #summary_matrix.toarray() #
  print(query_repr.shape)
  print("done with query repr")


  print("computed projection")
  print(proj_data.shape)
  word_vectors = proj_data.toarray()
  #compute_nearest_neighbors2(word_vectors, labels, "bulgaria")    
  #compute_nearest_neighbors2(word_vectors, labels, "diminish")
  #compute_nearest_neighbors2(word_vectors, labels, "dolophine")

  compute_nearest_neighbors3(query_repr, word_vectors, labels, "bulgaria")    
  compute_nearest_neighbors3(query_repr, word_vectors, labels, "diminish")
  compute_nearest_neighbors3(query_repr, word_vectors, labels, "dolophine")

  if False:

    def run_svd(neighbors):
      return la.svds(neighbors, k=100, ncv=None, tol=0, which='LM', v0=None, maxiter=None,return_singular_vectors=True)

    svd_result = define('svd_result', lambda: run_svd(proj_data), True)
    U, s, V = svd_result
    num_datapoints = U.shape[0]
    num_sing_v = U.shape[1]


  #zero the small components
    print("zero small projections")
    for i in range(0, num_sing_v):
      scores = list(U[:,i]) #this is the wrong 0 should be num_cols - 1
      ids = range(0, num_datapoints)

      sorted_values_ids = sorted(zip(ids,scores), key=lambda (_id,value): value)
      keep_top_k = 1000
      for (_id,v) in sorted_values_ids[keep_top_k:(num_datapoints - keep_top_k)]:
        #U[_id, i] = 0.0
        pass  

    compute_nearest_neighbors(U, s, labels, "bulgaria")  

def do_word_vectors2(script_vars):
  def define(var_name, fun, overwrite=False):
    if script_vars.has_key(var_name) and not overwrite:
      print('%s is already defined' % var_name)
      return script_vars[var_name]
    else:
      print('computing variables %s' % var_name)
      value = fun()
      script_vars[var_name] = value
      globals()[var_name] = value      
      return value

  print('svd on thresholded word vectors')
  word_vector_data = define('word_vector_data', lambda: read_neighbors(), True)
  ((rows, cols, scores), data, word2_id) = word_vector_data

  labels = [None]*len(word2_id)
  for (k,v) in word2_id.iteritems():
    labels[v] = k

  
  def run_svd(neighbors):
    return la.svds(data, k=500, ncv=None, tol=0, which='LM', v0=None, maxiter=None,return_singular_vectors=True)

  svd_result = define('svd_result', lambda: run_svd(neighbors), True)
  U, s, V = svd_result
  num_datapoints = U.shape[0]
  num_sing_v = U.shape[1]


#zero the small components
  print("zero small projections")
  for i in range(0, num_sing_v):
    scores = list(U[:,i]) #this is the wrong 0 should be num_cols - 1
    ids = range(0, num_datapoints)

    sorted_values_ids = sorted(zip(ids,scores), key=lambda (_id,value): value)
    keep_top_k = 1000
    for (_id,v) in sorted_values_ids[keep_top_k:(num_datapoints - keep_top_k)]:
      U[_id, i] = 0.0
      pass  

  compute_nearest_neighbors(U, s, labels, "bulgaria")

  compute_nearest_neighbors(U, s, labels, "diminish")

#neighbors of word bulgaria
#0)albania	0.0564146345065
#1)serbia	0.0559905431233
#2)bulgaria	0.0527730491904
#3)macedonia	0.0509880428797
#4)montenegro	0.0487048944796
#5)yugoslav	0.0478950452841
#6)kosovo	0.0419913493596
#7)yugoslavia	0.0406406959599
#8)albanian	0.0383340369025
#9)serbian	0.0382797233446



  #for l in labels:
  #  print l
  
  print("num sing. vectors in U " + str(num_sing_v))
  for i in range(0, 0):
    scores = list(U[:,num_sing_v -i - 1]) #
    #print(scores[0:10])
    ids = range(0, len(scores))
    print '---------------------------------'
    print ('cluster_' + str(i))
    sorted_values_ids = sorted(zip(ids,scores), key=lambda (_id,value): -value)
    for (_id,v) in sorted_values_ids[1:20]:
      lab = labels[_id]
      print("%f  %s" % (v, lab))

    print '---------------------------------'
    print ('cluster_' + str(i) + "+")
    sorted_values_ids = sorted(zip(ids,scores), key=lambda (_id,value): value)
    for (_id,v) in sorted_values_ids[1:20]:
      lab = labels[_id]
      print("%f  %s" % (v, lab))



def load_script(script_vars):
  def define(var_name, fun, overwrite=False):
    if script_vars.has_key(var_name) and not overwrite:
      print('%s is already defined' % var_name)
      return script_vars[var_name]
    else:
      print('computing variables %s' % var_name)
      value = fun()
      script_vars[var_name] = value
      globals()[var_name] = value      
      return value
  
    print(globals().keys())
    custom_data_home="/home/stefan2/mnistdata"

  custom_data_home="/home/stefan2/mnistdata"

  define('labels', lambda: load_labels())
  define('neighbors', lambda: load_neighbors(labels), True)
  print('loaded neighbors')
  
  #show_vector_plot(images[676,:])

  do_images = True
  if (do_images):
    define('images', lambda: load_images())
    initialize_cluster_centers(neighbors, labels, images)
  #do_word_vectors(script_vars)

  if False:
    def run_svd(neighbors):
      return la.svds(neighbors, k=50, ncv=None, tol=0, which='LM', v0=None, maxiter=None,return_singular_vectors=True)

    svd_result = define('svd_result', lambda: run_svd(neighbors), True)
    U, s, V = svd_result
    for l in labels:
      print l
    for i in range(0, 50):
      scores = list(U[:,i])
      #print(scores[0:10])
      ids = range(0, len(scores))
      print '---------------------------------'
      print ('cluster_' + str(i))
      sorted_values_ids = sorted(zip(ids,scores), key=lambda (_id,value): -value)
      for (_id,v) in sorted_values_ids[1:5]:
        lab = labels[_id]
        print("%f  %d %d" % (v, _id, lab))

      print '---------------------------------'
      print ('cluster_' + str(i) + "+")
      sorted_values_ids = sorted(zip(ids,scores), key=lambda (_id,value): value)
      for (_id,v) in sorted_values_ids[1:5]:
        lab = labels[_id]
        print("%f  %d %d" % (v, _id, lab))
    #plt.hist(scores, bins=50,alpha=0.5)
    #plt.grid(True)
    #plt.show()
    #plot(s)
    #show()
    #t = arange(0.0, 2.0, 0.01)
    #s = sin(2*pi*t)
    #plot(t, s)
    #show()

  

