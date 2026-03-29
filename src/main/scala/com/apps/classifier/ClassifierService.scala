package com.apps.classifier

import java.io.{File, PrintWriter}
import java.nio.file.{Path, Paths}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object ClassifierService {

  private def dataDir: Path = Paths.get(System.getProperty("user.dir"), "data")

  def main(args: Array[String]): Unit = {
    println("Start classifier")
    val train = readTrainData()

    val (bestKnn, foldSplits) = crossValidation(
      train,
      foldCount = 5,
      knnCandidates = Seq(1, 3, 5, 7, 9, 11, 15)
    )
    println(s"Выбранное K (число соседей) = $bestKnn")
    writeCvToFiles(foldSplits)

    val test = readTestData()
    val knnClassifier = new KnnClassifier(bestKnn, train, test)
    val result = knnClassifier.kNearestNeighbors
    writeLabelsToFile(result.map(_._2))
    println("Stop classifier")
  }

  /**
    * Подбор гиперпараметра K для k-NN по кросс-валидации.
    * Раньше число фолдов ошибочно передавалось в классификатор как K соседей,
    * а обучающая выборка для фолда содержала и тестовый фолд (`cTest ++ c`).
    */
  def crossValidation(
      train: Array[(Array[Char], Int)],
      foldCount: Int,
      knnCandidates: Seq[Int]
  ): (Int, Array[(Array[Int], Array[Int])]) = {
    require(train.length >= foldCount, "недостаточно примеров для кросс-валидации")
    require(foldCount > 1, "foldCount должен быть > 1")

    val n = train.length
    val foldSizes = splitIntoFoldSizes(n, foldCount)
    val foldRanges = foldRangesFromSizes(foldSizes)

    def evalKnn(knn: Int): (Double, Double) = {
      val accs = ArrayBuffer.empty[Double]
      for (foldIdx <- foldRanges.indices) {
        val testIdx = foldRanges(foldIdx)
        val testSet = testIdx.toSet
        val trainIdx = (0 until n).filterNot(testSet.contains).toArray

        val cTrain = trainIdx.map(train)
        val cTest = testIdx.map(train)

        val cvKnnClassifier = new KnnClassifier(knn, cTrain, cTest.map(_._1))
        val result = cvKnnClassifier.kNearestNeighbors
        accs += accuracyRate(result.map(_._2), cTest.map(_._2))
      }
      val mean = accs.sum / accs.length
      val variance = sampleVariance(accs.toArray)
      (mean, variance)
    }

    val scored = knnCandidates.map { knn =>
      val (mean, variance) = evalKnn(knn)
      println(f"K=$knn%2d: средняя точность=$mean%.4f, дисперсия по фолдам=$variance%.6f")
      (knn, mean, variance)
    }

    val bestKnn = scored.maxBy { case (_, mean, variance) => (mean, -variance) }._1

    val splits = foldRanges.map { testIdx =>
      val testSet = testIdx.toSet
      val trainIdx = (0 until n).filterNot(testSet.contains).toArray
      (trainIdx, testIdx)
    }

    (bestKnn, splits)
  }

  private def splitIntoFoldSizes(n: Int, foldCount: Int): Array[Int] = {
    val base = n / foldCount
    val rem = n % foldCount
    val sizes = Array.fill(foldCount)(base)
    var i = 0
    while (i < rem) {
      sizes(i) += 1
      i += 1
    }
    sizes
  }

  private def foldRangesFromSizes(sizes: Array[Int]): Array[Array[Int]] = {
    val out = Array.ofDim[Array[Int]](sizes.length)
    var start = 0
    var i = 0
    while (i < sizes.length) {
      val end = start + sizes(i)
      out(i) = (start until end).toArray
      start = end
      i += 1
    }
    out
  }

  private def sampleVariance(vals: Array[Double]): Double = {
    if (vals.length < 2) 0.0
    else {
      val m = vals.sum / vals.length
      vals.map(x => math.pow(x - m, 2)).sum / (vals.length - 1)
    }
  }

  def accuracyRate(predicted: Array[Int], actual: Array[Int]): Double = {
    require(predicted.length == actual.length)
    var correct = 0
    var i = 0
    while (i < predicted.length) {
      if (predicted(i) == actual(i)) correct += 1
      i += 1
    }
    correct.toDouble / predicted.length
  }

  def readTrainData(): Array[(Array[Char], Int)] = {
    val trainDir = dataDir.resolve("train").toFile
    val lblFile = dataDir.resolve("train").resolve("labels.txt").toFile
    val labels = Source.fromFile(lblFile, "UTF-8").getLines().map(_.trim.toInt).toArray

    val imgFiles = trainDir
      .listFiles()
      .filter(f => f.isFile && f.getName.endsWith(".pgm"))
      .sortBy(_.getName)

    require(
      imgFiles.length == labels.length,
      s"Число изображений (${imgFiles.length}) не совпадает с числом меток (${labels.length})"
    )

    imgFiles.zip(labels).map { case (img, lbl) => (readFile(img), lbl) }
  }

  def readTestData(): Array[Array[Char]] = {
    val testDir = dataDir.resolve("test").toFile
    require(testDir.isDirectory, s"Нет каталога: ${testDir.getPath}")

    val imgFiles = testDir
      .listFiles()
      .filter(f => f.isFile && f.getName.endsWith(".pgm"))
      .sortBy(_.getName)

    imgFiles.map(readFile)
  }

  def readFile(file: File): Array[Char] = {
    val src = Source.fromFile(file, "UTF-8")
    try src.toArray
    finally src.close()
  }

  def writeLabelsToFile(results: Array[Int]): Unit = {
    val outFile = dataDir.resolve("test").resolve("labels.txt").toFile
    val pw = new PrintWriter(outFile, "UTF-8")
    try results.foreach(r => pw.println(r.toString))
    finally pw.close()
  }

  def writeCvToFiles(results: Array[(Array[Int], Array[Int])]): Unit = {
    val outFileTrain = dataDir.resolve("train_indices_kfold.txt").toFile
    val outFileTest = dataDir.resolve("test_indices_kfold.txt").toFile

    val pwTrain = new PrintWriter(outFileTrain, "UTF-8")
    try results.foreach { case (trainInd, _) => pwTrain.println(trainInd.mkString(" ")) }
    finally pwTrain.close()

    val pwTest = new PrintWriter(outFileTest, "UTF-8")
    try results.foreach { case (_, testInd) => pwTest.println(testInd.mkString(" ")) }
    finally pwTest.close()
  }

}
