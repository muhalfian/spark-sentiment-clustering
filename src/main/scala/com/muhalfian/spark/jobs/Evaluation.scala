package com.muhalfian.spark.jobs

import com.muhalfian.spark.util._

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql._

import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import scala.collection.mutable.ArrayBuffer
import scala.util.{Success, Try}

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.mllib.util.MLUtils

object Evaluation extends StreamUtils {

  def main(args: Array[String]): Unit = {

    val SparkSession = getSparkSession(args)
    import SparkSession.implicits._


    val tweetsDF = SparkSession.read.json("/home/blade1/resentiment_agg.json")
    val tweetsDFTesting = SparkSession.read.json("/home/blade1/chandra_training_result.json")
    //val spark : SparkSession = SparkSession.builder.master("local[*]").getOrCreate

    val tweetsRDD = tweetsDF.rdd
    val TestingRDD = tweetsDFTesting.rdd
    //println(TestingRDD)
    // tweetsRDD.collect().foreach(println)

    val bagOfWord = tweetsRDD.map(
      row => {
        val label = row.getLong(0)
        val tweets = row.getString(2)
        (label, tweets.split(" ").toSeq)
      } 
    )

    val bagOfWordtest = TestingRDD.map(
      row => {
        val label = row.getLong(0)
        val tweets = row.getString(1)
        (label, tweets.split(" ").toSeq)
      } 
    )

    val splits = bagOfWord.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = bagOfWordtest


    val hashingTF = new HashingTF(2000)

    val training_labeled = training.map(
      t => (t._1, hashingTF.transform(t._2))
    ).map(
      x => new LabeledPoint((x._1).toDouble, x._2)
    )

    def time[R](block: => R): R={
      val t0 = System.nanoTime()
      val result = block
      val t1 = System.nanoTime()
      println("\n\nElapsed time: " + (t1 - t0)/1000000 + "ms")
      result
    }

    println(training_labeled)
    //training_labeled.collect().foreach(println)

    println("\n\n************** Training **************\n\n")

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
       .setNumClasses(3)
       .run(training_labeled)

    println("\n\n************** Testing **************\n\n")

    val predictionAndLabels = test.map(
      x => {
        val prediction = model.predict(hashingTF.transform(x._2))
        (prediction, x._1.toDouble)
     }
   )

  println(predictionAndLabels)
  //start evaluation with matric
  // Instantiate metrics object
  val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Overall Statistics
val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

   // 
//    val accuracy = 1.0 * predictionAndLabels.filter(x => x._1 == x._2).count() / test.count()
    
    println("Training and Testing Complete, accuracy is = " + accuracy)
    println("\nSome Predictions:\n")
  }
}