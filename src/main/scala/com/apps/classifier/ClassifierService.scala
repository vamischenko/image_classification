package com.apps.classifier

import java.io.{PrintWriter, File}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object ClassifierService {

  def main(args: Array[String]) = {
    println("Start classifier")
    val train = readTrainData()

    val cv = crossValidation(train)
    println(s"K = ${cv._1}")
    writeCvToFiles(cv._2)

    val test = readTestData()
    val knnClassifier = new KnnClassifier(cv._1, train, test)
    val result = knnClassifier.kNearestNeighbors
    writeLabelsToFile(result.map(x => x._2))
    println("Stop classifier")

  }

  def crossValidation(train: Array[(Array[Char], Int)]) = {
    val cvErrors = ArrayBuffer[(Int, Double)]()
    val cvTrainTestInc = ArrayBuffer[(Array[Int], Array[Int])]()
    var k = 10
    while (k > 1) {
      val kLenght: Int = train.length/k
      var cTrain = Array[(Array[Char], Int)]()
      var cTest = Array[(Array[Char], Int)]()
      val accs = ArrayBuffer[Double]() //на каждом этапе будем сохранять точность
      for (i <- 0 to k-1) {
        //промежуток длины kLenght берется в качестве тестовой выборки
        val testIndices = (i*kLenght to (i+1)*kLenght-1).toArray
        cTest = for (f <- testIndices) yield { train(f)}

        var trainIndices = Array[Int]()
        var inc1 = Array[Int]()
        var inc2 = Array[Int]()
        for (j <- 0 to i-1) {
          inc1 = inc1 ++ (j*kLenght to (j+1)*kLenght-1)
        }
        for (j <- i+1 to k-1) {
          inc2 = inc2 ++ (j*kLenght to (j+1)*kLenght-1)
        }
        trainIndices = inc1 ++ inc2

        val c = for (f <- trainIndices) yield { train(f)}
        cTrain = cTest ++ c

        val cvKnnClassifier = new KnnClassifier(k, cTrain, cTest.map(x => x._1))
        val result = cvKnnClassifier.kNearestNeighbors
        accs.append(calculateAccurace(result.map(x => x._2), cTest.map(x => x._2)))
        cvTrainTestInc.append((trainIndices, testIndices))
      }

      val cvError = getVariance(accs.toArray)
      cvErrors.append((k, cvError))
      k -= 1
    }
    cvErrors.foreach{x=>
      println(x._1 + " - " + x._2)
    }
    (cvErrors.minBy(_._2)._1, cvTrainTestInc.toArray)
  }

  def getVariance(vals: Array[Double]): Double = {
    val m = vals.sum/vals.length
    val d = vals.map(x => Math.pow(x - m, 2))
    d.sum*100/(vals.length - 1)
  }

  def calculateAccurace(res: Array[Int], labels: Array[Int]): Double = {
    var sum: Double = 0
    for (i <- res.indices) {
      if (res(i) == labels(i)) {
        sum += 1
      }
    }
    sum
  }


  def readTrainData() = {
    val absPath = new File(".").getAbsolutePath
    val trainDataPath = new File(absPath.substring(0, absPath.length-1) + "/data/train")
    val lblFile = new File(absPath.substring(0, absPath.length-1) + "/data/train/labels.txt")
    val labels = Source.fromFile(lblFile).map(_.toString).filter(s => s.indexOf("\n") < 0).toList

    val imgFiles = trainDataPath.listFiles().filter(file => file.getName.indexOf("labels.txt") < 0)
    val train = ArrayBuffer[(Array[Char], Int)]()
    var i = 0
    for (img <- imgFiles) {
      val imgPoints = readFile(img)
      train.append((imgPoints, labels(i).toInt))
      i += 1
    }
    train.toArray
  }

  def readTestData() = {
    val absPath = new File(".").getAbsolutePath
    val trainDataPath = new File(absPath.substring(0, absPath.length-1) + "/data/test")
    val imgFiles = trainDataPath.listFiles()
    val test = new ArrayBuffer[Array[Char]]()
    for (img <- imgFiles) {
      test.append(readFile(img))
    }
    test.toArray
  }

  def readFile(file: File) = {
    val imgPoints = scala.io.Source.fromFile(file)
    imgPoints.toArray
  }

  def writeLabelsToFile(results: Array[Int]) = {
    val absPath = new File(".").getAbsolutePath
    val outFile = new File(absPath.substring(0, absPath.length-1) + "/data/test/labels.txt")
    val pw = new PrintWriter(outFile)
    for (res <- results) {
      pw.write(res.toString + "\n")
    }
    pw.close()

  }

  def writeCvToFiles(results: Array[(Array[Int], Array[Int])]) = {
    val absPath = new File(".").getAbsolutePath
    val outFileTrain = new File(absPath.substring(0, absPath.length-1) + "/data/train_indices_kfold.txt")
    val outFileTest = new File(absPath.substring(0, absPath.length-1) + "/data/test_indices_kfold.txt")

    val pwTrain = new PrintWriter(outFileTrain)
    for (trainInd <- results.map(x => x._1)) {
      pwTrain.write(trainInd.mkString(" ") + "\n")
    }
    pwTrain.close()

    val pwTest = new PrintWriter(outFileTest)
    for (testInd <- results.map(x => x._2)) {
      pwTest.write(testInd.mkString(" ") + "\n")
    }
    pwTest.close()

  }

}
