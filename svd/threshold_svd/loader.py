import sys
import threshold_svd
import sys, traceback

mem = {}

while True:
    try:
      print('loading script')
      reload(threshold_svd)
      threshold_svd.load_script(mem)
    except:
      print "Exception in user code:"
      print '-'*60
      traceback.print_exc(file=sys.stdout)
      print '-'*60
    print "Press enter to re-run the script, CTRL-C to exit"
    sys.stdin.readline()

