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
  ids = np.random.choice(n, 2**16,replace=False) #needs to be a power of 2 and less than n
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
  n,m = data.shape
  print("num data points %s" % n)
  #run the method after successive othogonalization
  for j in range(0, 50):
    print("iteration: " + str(j))
    res = find_dominant_directions(data)
    plot_vector_png("pattern_" + str(j), res)
    for i in range(0, n):
      v = data[i,:]
      proj = np.reshape(v, (1, m)).dot(np.reshape(res, (m,1)))[0,0]
      data[i,:] = v - proj*res


  

