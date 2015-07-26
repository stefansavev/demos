
import os
from collections import defaultdict

by_label = defaultdict(lambda: [])

for dirpath, dnames, fnames in os.walk("./"):
    for f in fnames:
        if f.endswith(".png"):
            #f is for example fig_3_1_94.png
            (fig, label, index, freq) = f.replace(".png", "").split("_")
            by_label[int(label)].append( (index, freq, f ) )

max_len = max(map(lambda values: len(values), by_label.values()))

print("<html><body>")
print("<table>")
print("<tbody>")

for (label, values) in by_label.iteritems():
  print("<tr>")
  print('  <th>%d</th>' % label)
  s = sorted(values, key = lambda (index,freq,f): index)
  for (index,freq,f) in s:
    print('  <th><img class="center" src="/assets/images/svd/mnist-stagewise-patterns/%s" width="50"></th>' % f)
    #print('  <th><img class="center" src="%s" width="50"></th>"' % f)
  for i in range(len(s), max_len):
    print(' <th></th>')  
  print("</tr>")
  
print("</tbody>")
print("</table>")
print("</body></html>")
  

