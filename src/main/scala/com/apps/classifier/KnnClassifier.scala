package com.apps.classifier

import scala.collection.mutable.ArrayBuffer

class KnnClassifier(kNN: Int, train: Array[(Array[Char], Int)], test: Array[Array[Char]]) {


  def kNearestNeighbors: Array[(Array[Char], Int)] = {
    val result = ArrayBuffer[(Array[Char], Int)]()
    for (testImg <- test) {
      val disort = {
        for (t <- train.indices) yield { (distance(testImg, train(t)._1), train(t)._2) }
      }.sortBy(_._1)

      val partdisort = disort.take(kNN)
      val highest = highestMultipleFrequency(partdisort)
      highest match {
        case x: Some[(Int, Int)] => {
          result.append((testImg, x.get._2))
        }
        case _ => {
          result.append((testImg, -1))
        }
      }
    }
    result.toArray
  }

  def printImage(img: Array[Char]) {
    for (x <- 0 until 28) {
      println()
      for (y <- 0 until 28)
        print(img(x * 28 + y) + " ")
    }
    println()
  }

  def highestMultipleFrequency[T](items: IndexedSeq[T]): Option[T] = {
    type Frequencies = Map[T, Int]
    type Frequency = Pair[T, Int]

    def freq(acc: Frequencies, item: T) = acc.contains(item) match {
      case true => acc + Pair(item, acc(item) + 1)
      case _ => acc + Pair(item, 1)
    }
    def mostFrequent(acc: Option[Frequency], item: Frequency) = acc match {
      case None if item._2 >= 0 => Some(item)
      case Some((value, count)) if item._2 > count => Some(item)
      case _ => acc
    }
    items.foldLeft(Map[T, Int]())(freq).foldLeft[Option[Frequency]](None)(mostFrequent) match {
      case Some((value, count)) => Some(value)
      case _ => None
    }
  }

  def distance(a: Array[Char], b: Array[Char]): Int = {
    var sum = 0
    for (i <- a.indices) {
      if (a(i) != b(i)) {
        sum += 1
      }
    }
    sum
  }

}
