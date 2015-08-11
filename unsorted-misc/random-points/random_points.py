from sklearn.datasets import fetch_mldata
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib
import pylab as P

def show_vector_plot(flattened_image):
    image = np.reshape(flattened_image, (-1, 28))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show() 

def pick_random_points(available_ids):
  n = len(available_ids)  
  (x_index, y_index) = np.random.choice(n, 2, replace=False)
  x_id = available_ids[x_index]
  y_id = available_ids[y_index]
  return (x_id, y_id)

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

def majority_label(labels):
  d = {}
  for k in labels:
    if d.has_key(k):
      d[k] += 1        
    else:
      d[k] = 1
          
  top = sorted(d.iteritems(), key = lambda (k,v): -v)
  print(str(top[0]))
  return top[0]

def pick_random_pattern(stage, available_ids, prev_dir, data, labels):
  print("\n\n-----------------------------------")
  print("Stage: " + str(stage))

  print("Len ids: " + str(len(available_ids)))  
  selected_labels = []
  for _id in available_ids:
    selected_labels.append(labels[_id])
  (k,v) = majority_label(selected_labels)    
  print("majority: " + str((k, 100.0*v/len(available_ids))))

  (x_id, y_id) = pick_random_points(available_ids)
  pnt_x = data[x_id,:]
  pnt_y = data[y_id,:]
  plot_vector_png("stage_" + str(stage) + "_x", pnt_x)
  plot_vector_png("stage_" + str(stage) + "_y",pnt_y)
  new_dir = pnt_x - pnt_y
  new_dir /= linalg.norm(new_dir)  
  plot_vector_png("stage_" + str(stage) + "_x_min_y_", new_dir)
  prev_dir_norm = linalg.norm(prev_dir)
  if (prev_dir_norm > 0.001):
    print("normalzing prev dir")
    prev_dir /= prev_dir_norm 

  print("new dir shape " + str(new_dir.shape))
  print("prev dir shape: " + str(prev_dir.shape))
  m = prev_dir.shape[0]
  proj_new_dir = np.reshape(new_dir, (1, m)).dot(np.reshape(prev_dir, (m,1)))[0,0]

  print("proj " + str(proj_new_dir))
  print("norm before proj: " + str(linalg.norm(new_dir)))
  print("norm prev dir: " + str(linalg.norm(prev_dir)))
  new_dir = new_dir - proj_new_dir*prev_dir

  print("norm after proj: " + str(linalg.norm(new_dir)))
  norm = linalg.norm(new_dir)
  new_dir /= norm
  #show_vector_plot(new_dir)
  plot_vector_png("stage_" + str(stage) + "_x_min_y_norm_ortho", new_dir)

  #project the complete dataset on the pattern
  proj = data.dot(new_dir.T)

  fig =P.figure()

  # the histogram of the data with histtype='step'
  n1, bins, patches = P.hist(proj, 50, normed=1, histtype='bar', rwidth=0.8)

  #P.show()
  fig.savefig("stage_" + str(stage) + "_hist.png", dpi=200)
  s = set(available_ids)
  zipped = filter(lambda (v,i): i in s, zip(list(proj), range(0, len(proj))))

  sorted_zipped = sorted(zipped, key = lambda (v,i): v) #sort by overlap
  
  selected_labels = map(lambda (v,i): labels[i], sorted_zipped)[0:99]
  (k,v) = majority_label(selected_labels)    
  print("majority in top 100 " + str((k,v)))
    
  sorted_zipped = sorted(zipped, key = lambda (v,i): -v) #sort by overlap
  
  selected_labels = map(lambda (v,i): labels[i], sorted_zipped)[0:99]
  (k,v) = majority_label(selected_labels)    
  print("majority in top 100 " + str((k,v)))

  mean = sum(map(lambda (value,_id): value, zipped))/float(len(zipped))
  print(str(stage) + " mean " + str(mean))  
  #mean = 0.0  
  pos = filter(lambda (value, _id): value >= mean, zipped)
  neg = filter(lambda (value, _id): value < mean, zipped)
  print(str(stage) + " pos len " + str(len(pos)))
  print(str(stage) + " neg len " + str(len(neg)))
  pos_ids = map(lambda (value,_id): _id, pos)
  neg_ids = map(lambda (value,_id): _id, neg)
  return (pos_ids, neg_ids, new_dir)  
  #print(pos_ids[0:10])
  

def create_patterns(input_data, input_labels):
  n, m = input_data.shape
  all_ids = range(0, n)
  prev_dir = np.ndarray(shape=(1, m), dtype=float, order='F')[0]  
  for i in range(0, m):
    prev_dir[i] = 0.0 
 
  for i in range(0, 10):
    (pos_ids, neg_ids, new_dir) = pick_random_pattern(i, all_ids, prev_dir, input_data, input_labels) 
    all_ids = pos_ids
    prev_dir = new_dir  
    #print("second pattern")
    #pick_random_pattern(1, pos_ids, prev_dir, input_data, input_labels) 

def combine(vecs):
  mid = len(vecs)/2  
  while (mid > 1):
    for i in range(0, mid):
      v1 = vecs[i]
      v2 = vecs[i + mid]
      v_plus = v1 + v2
      v_minus = v1 - v2
      v_res = None    
      if (linalg.norm(v_plus) >= linalg.norm(v_minus)):
        v_res = v_plus
        #print "+"        
      else:
        v_res = v_minus
        #print "-"
      vecs[i] = v_res
    mid = mid /2
  return vecs[0]
      
def find_dominant_directions(data):
  n, m = data.shape
  ids = np.random.choice(n, 256*16,replace=False)
  vecs = []
  for _id in ids:
    vecs.append(data[_id,:])
  res = combine(vecs) 
  res = res / linalg.norm(res) 
  return res

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
  data = mnist.data.astype(float) #[0:100,:] #convert to float
  labels = mnist.target #[0:100]
  print "ready"  
  #create_patterns(data, labels)
  #res = find_dominant_directions(data)
  n,m = data.shape
  #show_vector_plot(res)

  for j in range(0, 50):
    print(str(j))
    res = find_dominant_directions(data)
    #show_vector_plot(res)
    plot_vector_png("pattern_" + str(j), res)
    for i in range(0, n):
      v = data[i,:]
      proj = np.reshape(v, (1, m)).dot(np.reshape(res, (m,1)))[0,0]
      #print(proj)
      data[i,:] = v - proj*res


  

