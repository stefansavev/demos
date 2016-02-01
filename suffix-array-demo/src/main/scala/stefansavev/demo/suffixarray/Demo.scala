package stefansavev.demo.suffixarray

package stefansavev.demo.hyperloglog.similarity

import java.io.{BufferedInputStream, File, FileInputStream}
import java.nio.file.{StandardOpenOption, Paths, Files}

import scala.collection.mutable.ArrayBuffer

object TwentyNewsgroupsNgrams {

  def readFileAsString(fileName: String, howManyMB: Int = 128): String = {
    println("Reading files")
    val file = new BufferedInputStream(new FileInputStream(new File("c:/20NGExcercise/singlefile.txt")))
    val buffer = Array.ofDim[Byte](1024*1024*howManyMB) //file is max 128 MB
    val len = file.read(buffer)
    val chars = buffer.take(len).map(b => b.toChar)
    new String(chars)
  }

  def tokenizeString(str: String): Array[String] = {
    println("Tokenizing file contents")
    val matches = """[a-zA-Z0-9]+""".r.findAllIn(str)
    matches.map(_.toLowerCase()).toArray
  }


  def main(args: Array[String]): Unit = {
    //Download 20 newsgroups dataset and
    //use the command to put the text in a single file
    //find . -type f  -exec cat {} \+ > ../singlefile.txt
    val fileName = "C:/20NGExcercise/singlefile.txt"
    val contents = readFileAsString(fileName, 128)
    val words = tokenizeString(contents)

    val s2id = scala.collection.mutable.HashMap[String, Int]()
    val id2s = scala.collection.mutable.HashMap[Int, String]()
    val wordCTF = new ArrayBuffer[Int]()
    wordCTF += 0 //the word ids start at 1

    def add(s: String): Int = {
      //smallest wordId must be 1
      if (s2id.contains(s)){
        val id = s2id(s)
        wordCTF(id) += 1
        id
      }
      else{
        val cnt = s2id.size + 1
        s2id += ((s, cnt))
        id2s += ((cnt, s))
        wordCTF += 0 //extend by one position
        wordCTF(cnt) = 1
        cnt
      }
    }

    val wordIds = words.map(s => add(s))
    val pattern = Array("colorado", "edu", "organization", "university")

    var counter = 0

    def incCounter(window: Array[String]): Unit = {
      if (window.length == 4 &&
           window(0) == pattern(0)
        && window(1) == pattern(1)
        && window(2) == pattern(2)
        && window(3) == pattern(3)){
        counter += 1
      }
    }

    words.sliding(4).foreach(incCounter)
    println("Count for pattern: " + counter) //should be 199

    val ngramAnalyzer = new NGramAnalyzer(wordIds, i => id2s(i))
    ngramAnalyzer.dumpNgrams(4, 50, wordCTF.toArray)
  }
}

