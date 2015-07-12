from scipy.sparse import csr_matrix
import numpy as np
import scipy
import scipy.sparse.linalg as la

file_name = "D:/Kaggle/20Newsgroups/20newsgroups-words.tsv"

f = open(file_name, 'r')
docids = {} #index -> doc_id
rows = [] #document ids
cols = [] #word ids
scores = [] #score for (doc_id, word_id)
words = [] #list of words indexed by their id
word_to_id = {} #word string to id

row_id = 0
for line in f:
    features = line.split("\t")
    word = features[0].split(":")[1]
    word_id = len(words)
    words.append(word)
    word_to_id[word] = word_id
    
    #optinally normalize the input data
    norm = 0.0
    for i in range(2, len(features)):
        parts = features[i].split(":")
        score = float(parts[1])
        norm += score #*score
        
    #norm = norm + 1.0 #np.sqrt(norm + 10.0)    
    for i in range(2, len(features)):
        parts = features[i].split(":")
        docid = parts[0]
        score = float(parts[1])        
        if (not docids.has_key(docid)):
            docids[docid] = len(docids) 
        rows.append(row_id)
        cols.append(docids[docid])
        #score = np.sqrt(score/norm) #do not normalize (performs worse)
        scores.append(score)
    row_id += 1
f.close()

data = csr_matrix((scores, (rows, cols)), shape=(row_id, len(docids)))
print("shape of input data (num_docs * num_words) " + str(data.shape))

#compute top 100 singular vectors
U, s, V = la.svds(data.T, k=100, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True)
print("shape of U " + str(U.shape))
print("shape of V " + str(V.shape))

print_top_words_for_each_V = True
plot_U = True
plot_V = True
write_most_similar_words_examples = True

if print_top_words_for_each_V:
    for r in range(0, 100):
        row = V.T[:,r]
        i = 0
        id_and_values = []
        for value in row:
            id_and_values.append((i, value))
            i += 1

        print "-------------------"
        print('i = ' + str(r) + " -")            
        sorted_id_and_values = sorted(id_and_values, key = lambda (k,v): v)
        for i in range(0, 50):
            (k,v) = sorted_id_and_values[i]
            print(words[k] + "\t" + str(v))
        
        sorted_id_and_values = sorted(id_and_values, key = lambda (k,v): -v)
        print "-------------------"
        print('i = ' + str(r) + " +")
        for i in range(0, 50):
            (k,v) = sorted_id_and_values[i]
            print(words[k] + "\t" + str(v))    
    
       
import matplotlib.pyplot as plt

if plot_U:   
    for r in range(0, 100):
        row_u = U[:,r]
        print("length of row_u " + str(len(row_u)))
        plt.hist(row_u, bins=50,alpha=0.5)
        plt.grid(True)
        #plt.show()
        plt.savefig("u_" + str(r) + ".png", bbox_inches='tight')
        plt.close()
      
if plot_V:   
    for r in range(0, 100):
        row_v = V[:,r]
        print("length of row_v " + str(len(row_v)))
        plt.hist(row_v, bins=50,alpha=0.5)
        plt.grid(True)
        #plt.show()
        plt.savefig("v_" + str(r) + ".png", bbox_inches='tight')
        plt.close()
        
if write_most_similar_words_examples:
    Sinv = np.diag(s/(s + 1))
    word_vectors = V.T.dot(Sinv)
    norms = []
    num_words = word_vectors.shape[0]
    for i in range(0, num_words):
        v = word_vectors[i, :]
        norms.append(np.sqrt(v.dot(v.T)))
            
    def word_nearest_neighbors(word):
        word_id = word_to_id[word]
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
            print(str(i) + ")" + words[k] + "\t" + str(v))  
            
    word_nearest_neighbors('windows') #('stopping')
    word_nearest_neighbors('jesus') #('stopping')
    word_nearest_neighbors('congress') #('stopping')
    
            