from sklearn.datasets import fetch_mldata
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib

#This script is used in conjuction with the loader.py script to be run in the interperter
#Just do:
# python
# >>import loader
# variables are cached and on modification of this script just press enter (Ctrl + C) to break

def sample_data(data, labels, m):
  n = data.shape[0]
  random_ids = np.random.choice(n, m, replace=False)
  new_data = np.ndarray(shape=(m, data.shape[1]), dtype=float, order='F')  
  new_labels = [None]*m  
  i = 0
  for _id in random_ids:
    new_data[i,:] = data[_id,:]
    new_labels[i] = labels[_id]    
    i += 1
  return new_data,new_labels

def show_vector_plot(flattened_image):
    image = np.reshape(flattened_image, (-1, 28))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def plot_vector_png(clazz, cnt, quality, flattened_image):
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(0.3, 0.3)

    image = np.reshape(flattened_image, (-1, 28))

    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

    filename = "fig_" + str(clazz) + "_" + str(cnt) + "_" + str(quality) + ".png" 
    fig.savefig(filename, dpi=100)


def build_residual_set(data, labels, point_names, U_input, V_input, new_features, next_feature_id, feature_dict, k, top):
  results = []
  selected_ids = {}  
  k = min(k, U_input.shape[1]) #may be U has fewer columns
    
  for i in range(0, k):
    #print("==========" + str(i))  
    u = U_input[:, i]
    label_value = zip(labels, u)
    feature_pattern = V_input[i,:]
    sorted_label_value1 = sorted(label_value, key = lambda (l,v): v)
    sorted_label_value2 = sorted(label_value, key = lambda (l,v): -v)
    id_value = zip(range(0, len(u)), u)
    sorted_id_value1 = sorted(id_value, key = lambda (_id,v): v)
    sorted_id_value2 = sorted(id_value, key = lambda (_id,v): -v)
    for (new_feature_id, sorted_id_value, sorted_label_value) in [(next_feature_id[0], sorted_id_value1, sorted_label_value1), (next_feature_id[0] + 1, sorted_id_value2, sorted_label_value2)]:
      maj_lab = majority_label(sorted_label_value)
      feature_dict[new_feature_id] = (maj_lab, feature_pattern)        
      for (key,v) in sorted_id_value[0:(top - 1)]:
        name = point_names[key] #point name
        new_features.append((name, new_feature_id, v))          
        if selected_ids.has_key(key):
          selected_ids[key] += 1
        else:
          selected_ids[key] = 1
    next_feature_id[0] += 2  
  num_selected_ids = data.shape[0] - len(selected_ids)
  print("num selected ids: %d" % num_selected_ids)
  m = data.shape[1]
  new_data = np.ndarray(shape=(num_selected_ids, m), dtype=float, order='F')  
  new_labels = [None]*num_selected_ids
  new_ids = [None]*num_selected_ids  
  next_id = 0  
  for _id in range(0, data.shape[0]): #sorted(selected_ids.keys()):
    if not selected_ids.has_key(_id):    
      new_data[next_id,:] = data[_id]
      new_labels[next_id] = labels[_id]
      new_ids[next_id] = point_names[_id]
      next_id += 1
  print('next id %d' % next_id)
  return (new_data, new_labels, new_ids)

def build_inv_index(inp):
  index = {}
  for (pointid, featureid, value) in inp:
    if index.has_key(pointid):
      index[pointid].append((featureid,value))
    else:
      index[pointid]= [(featureid,value)]
  return index

def build_direct_index(inp):
  index = {}
  for (pointid, featureid, value) in inp:
    if index.has_key(featureid):
      index[featureid].append((pointid,value))
    else:
      index[featureid]= [(pointid,value)]

def index_stats(index):
  total = 0
  i = 0
  for (k,v) in index.iteritems():
    total += len(v)
    i += 1
  print("index: avg list size %f" % (total/i))

def build_forest(input_data, input_labels):
  datai = input_data
  labelsi = input_labels
  point_names_i = range(0, len(input_labels))
  all_trees = []  
  new_features = []
  next_feature_id = [0]
  i = 0
  feature_dict = {}
  while(datai.shape[0] >= 1000 and i < 15):
    print("Iter %d (num data points %d)" % (i, datai.shape[0]))    
    Ui, si, Vhi = linalg.svd(datai, full_matrices=False)
    treei = get_top_labels((Ui,labelsi), 50)
    all_trees = all_trees + treei
    max_freq_per_label(all_trees)

    datai, labelsi, point_namesi = build_residual_set(datai, labelsi, point_names_i, Ui, Vhi, new_features, next_feature_id, feature_dict, 50, 100)

    print("num new features: %d" % len(new_features))
    print("feature dict: %d" % len(feature_dict))    
    inv_index = build_inv_index(new_features)
    index_stats(inv_index)    
    i += 1
  return (new_features, feature_dict)

def get_top_labels((U_input, labels), k):
  results = []
  for i in range(0, k):
    #print("==========" + str(i))  
    u = U_input[:, i]
    label_value = zip(labels, u)
    sorted_label_value1 = sorted(label_value, key = lambda (l,v): v)
    sorted_label_value2 = sorted(label_value, key = lambda (l,v): -v)
    r1 = majority_label(sorted_label_value1)
    r2 = majority_label(sorted_label_value2)
    results.append(r1)
    results.append(r2)
  return results

def majority_label(sorted_values):
  top_labels = map(lambda (k,v): k, sorted_values[0:99])
  d = {}
  for k in top_labels:
    if d.has_key(k):
      d[k] += 1        
    else:
      d[k] = 1
          
  top = sorted(d.iteritems(), key = lambda (k,v): -v)
  #print(str(top[0]))
  return top[0]

def max_freq_per_label(kvs):
  d = {}
  for (k,v) in kvs:
    if (d.has_key(k)):
      d[k] = max(d[k], v) 
    else:
      d[k] = v
  for (k,maxf) in d.iteritems():
    print((k,maxf))

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

  define('mnist', lambda: fetch_mldata('MNIST original', data_home=custom_data_home))
  data = mnist.data
  labels = mnist.target
  #data,labels = sample_data(data, labels, 25000)
  force_rebuild = False
  define('new_features', lambda: build_forest(data, labels), force_rebuild)
  ignore,feature_dict = script_vars["new_features"]
  
  s = sorted(feature_dict.values(), key = lambda ((lab,freq), pattern): -freq)


  import datetime
  import numpy as np
  from matplotlib.backends.backend_pdf import PdfPages
  import matplotlib.pyplot as plt

  for clazz in range(0, 10):
    f = filter(lambda ((lab,freq), pat): lab == clazz, s)
    f = f[0: min(len(f),10)]
    cnt = 1
    for ((lab, freq),pat) in f:
      print("plot figure ")
      print((lab,freq))
      plot_vector_png(clazz, cnt, freq, pat)
      cnt += 1




  

