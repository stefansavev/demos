import sys
import random_points
import sys, traceback

mem = {}

while True:
    try:
      print('loading script')
      reload(random_points)
      random_points.load_script(mem)
    except:
      print "Exception in user code:"
      print '-'*60
      traceback.print_exc(file=sys.stdout)
      print '-'*60
    print "Press enter to re-run the script, CTRL-C to exit"
    sys.stdin.readline()

