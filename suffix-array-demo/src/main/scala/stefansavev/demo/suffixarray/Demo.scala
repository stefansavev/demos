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

/*
Output is extracted 4grams that appear more than 50 times
       and are sorted by frequency

ax ax ax ax	90464.0
ustar jrennie 0000000 0000000	18888.0
0 ustar jrennie 0000000	18846.0
jrennie 0000000 0000000 from	18038.0
max ax ax ax	8276.0
ax max ax ax	8210.0
ax ax ax max	8186.0
ax ax max ax	8122.0
20news bydate train comp	2941.0
20news bydate train rec	2393.0
20news bydate train sci	2377.0
i don t know	2114.0
20news bydate test comp	1960.0
20news bydate train talk	1956.0
3 q 3 q	1748.0
world nntp posting host	1594.0
distribution world nntp posting	1594.0
20news bydate test rec	1594.0
20news bydate test sci	1583.0
bydate train talk politics	1578.0
q 3 q 3	1558.0
tin version 1 1	1426.0
x newsreader tin version	1426.0
newsreader tin version 1	1426.0
i don t think	1360.0
20news bydate test talk	1305.0
bydate train rec sport	1199.0
bydate train comp sys	1170.0
r g r g	1164.0
edu organization university of	1101.0
g9v g9v g9v g9v	1078.0
comp os ms windows	1063.0
bydate test talk politics	1053.0
comp sys ibm pc	1044.0
sys ibm pc hardware	1030.0
g r g r	1024.0
i would like to	1016.0
 */
