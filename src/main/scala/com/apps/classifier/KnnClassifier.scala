package com.apps.classifier

import scala.collection.mutable.ArrayBuffer

class KnnClassifier(kNN: Int, train: Array[(Array[Char], Int)], test: Array[Array[Char]]) {

  require(kNN > 0, "kNN must be positive")

  def kNearestNeighbors: Array[(Array[Char], Int)] = {
    val result = ArrayBuffer.empty[(Array[Char], Int)]
    for (testImg <- test) {
      val sorted =
        (for (t <- train.indices) yield (distance(testImg, train(t)._1), train(t)._2)).sortBy(_._1)
      val neighbors = sorted.take(kNN)
      val labels = neighbors.map(_._2)
      result.append((testImg, majorityVote(labels)))
    }
    result.toArray
  }

  def printImage(img: Array[Char]): Unit = {
    for (x <- 0 until 28) {
      println()
      for (y <- 0 until 28)
        print(s"${img(x * 28 + y)} ")
    }
    println()
  }

  /** Мажоритарное голосование по меткам соседей (раньше ошибочно считалась частота пар (расстояние, метка)). */
  private def majorityVote(labels: Seq[Int]): Int =
    labels.groupBy(identity).maxBy(_._2.size)._1

  def distance(a: Array[Char], b: Array[Char]): Int = {
    var sum = 0
    val n = math.min(a.length, b.length)
    var i = 0
    while (i < n) {
      if (a(i) != b(i)) sum += 1
      i += 1
    }
    sum + math.abs(a.length - b.length)
  }

}
