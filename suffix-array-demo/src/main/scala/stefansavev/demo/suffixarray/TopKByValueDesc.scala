package stefansavev.demo.suffixarray

import java.util.PriorityQueue

class TopKByValueDesc(val k: Int){

  class MinPointScore(var pointId: Int, var score: Double) extends Comparable[MinPointScore] {
    override def compareTo(o: MinPointScore): Int = {
      this.score.compareTo(o.score)
    }
    def set(pointId: Int, score: Double): Unit = {
      this.pointId = pointId
      this.score = score
    }
  }

  val priorityQueue = new PriorityQueue[MinPointScore]()

  def updateMaxPQ(pq: PriorityQueue[MinPointScore], k: Int, pntId: Int, score: Double): Unit ={
    if (pq.size() < k){
      pq.add(new MinPointScore(pntId, score))
    }
    else{
      val minAcceptedScore = pq.peek().score
      if (score > minAcceptedScore){
        val old = pq.remove() //remove min
        old.set(pntId, score)
        pq.add(old)
      }
    }
  }

  def addKeyValue(key: Int, value: Double): Unit = {
    updateMaxPQ(priorityQueue, k, key, value)
  }

  def getSortedAndReset(): (Array[Int], Array[Double]) = {
    val n = priorityQueue.size()
    val keys = Array.ofDim[Int](n)
    val values = Array.ofDim[Double](n)
    var i = 0
    while(i < n){
      val item = priorityQueue.remove()
      keys(n - i - 1) = item.pointId
      values(n - i - 1) = item.score
      i += 1
    }
    (keys, values)
  }
}
