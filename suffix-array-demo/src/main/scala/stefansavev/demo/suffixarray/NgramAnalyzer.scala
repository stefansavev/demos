package stefansavev.demo.suffixarray

import sais.SuffixArraySort

class NGramAnalyzer(input: Array[Int], int2String: Int => String, outputSep: String = " "){
  val inputLength = input.length

  private def length(index: Int) = {
    inputLength - index
  }

  private def charAt(index: Int, depth: Int): Int = {
    input(index + depth)
  }

  private def computeLCP(sa: Array[Int]): Array[Int] = {
    val n = sa.length
    val rank = Array.ofDim[Int](sa.length)
    val lcp = Array.ofDim[Int](sa.length)
    var i = 0
    while(i < n){
      rank(sa(i)) = i
      i += 1
    }

    var m = 0
    i = 0
    while(i < n - 1){
      val k = rank(i)
      if (k > 0) {
        val j = sa(k - 1)
        while (i + m < inputLength && j + m < inputLength && input(i + m) == input(j + m)) {
          m += 1
        }
        lcp(k) = m
        if (m > 0) {
          m -= 1
        }
      }
      i += 1
    }
    lcp
  }

  private var lcp: Array[Int] = null
  private var suffixArray: Array[Int] = null

  //call sort
  sort()

  def timed[R](msg: String, codeBlock: => R): R = {
    val start = System.currentTimeMillis()
    val result = codeBlock    // call-by-name
    val end = System.currentTimeMillis()
    val elapsed = end - start
    val timeAsStr = if (elapsed >= 1000) (elapsed/1000.0 + " secs.") else (elapsed + " ms.")
    println(s"Time for '${msg}' ${timeAsStr}")
    result
  }

  private def sort(): Unit = {
    val idLimit = input.max + 1
    for(i <- 0 until 4) { //repeat to get more reliable timings
      suffixArray = null
      suffixArray = timed("suffix array sort", SuffixArraySort.sort(this.input, idLimit))
      timed("adapted string sort", AdaptedStringSort.sort(this.input))
    }
    lcp = computeLCP(suffixArray)
  }

  private def computeScore(startOffset: Int, endOffset: Int, count: Int, wordScores: Array[Int]): Double = {
    count
    /*
    var i = startOffset
    var baseline = 0.0
    while(i < endOffset){
      val wordId = input(i)
      val wordScore = wordScores(wordId)
      baseline += Math.log(wordScore + 1.0)
      i += 1
    }
    val model = Math.log(count + 1.0)
    count*(model - 1.0*baseline)
    */
  }

  def dumpNgrams(ngram: Int, threshold: Int, wordScores: Array[Int] = null): Unit = {
    val k = threshold max 2
    var i = 1
    var cntr = 0
    val topK = new TopKByValueDesc(500)

    while(i <= lcp.length){
      val c = if (i < lcp.length) lcp(i) else 0
      if (c >= ngram){
        cntr += 1
      }
      else{
        cntr += 1
        if (cntr >= k) {
          val offset = suffixArray(i - 1)
          val score = computeScore(offset, offset + ngram, cntr, wordScores)
          topK.addKeyValue(offset, score)
        }
        cntr = 0
      }
      i += 1
    }
    val (topSortedKeys, topSortedValues) = topK.getSortedAndReset()
    val topSortedKeysStr = topSortedKeys.map(offset => input.slice(offset, offset + ngram).map(v => int2String(v)).mkString(outputSep))
    topSortedKeysStr.zip(topSortedValues).foreach({case (k,v) => {
      println(k + "\t" + v)
    }})
  }
}