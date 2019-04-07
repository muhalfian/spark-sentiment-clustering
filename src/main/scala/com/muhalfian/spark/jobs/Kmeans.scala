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

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}

object Kmeans extends StreamUtils {

  def main(args: Array[String]): Unit = {

    val SparkSession = getSparkSession(args)
    import SparkSession.implicits._


    val tweetsDF = SparkSession.read.json("/home/blade1/Documents/spark-sentiment-clustering/db/resentiment_agg.json")

    val tweetsRDD = tweetsDF.rdd

    val rawTweets = tweetsRDD.map(
      row => {
        val label = row.getLong(0)
        val tweets = row.getString(2)
        (label, tweets.split(" ").toSeq)
      }
    )
 
    // val splits = rawTweets.randomSplit(Array(0.8, 0.2), seed = 11L)
    // val training = splits(0).cache()
    // val test = splits(1).cache()

    val hashingTF = new HashingTF(2000)

    // val vectors = rawTweets.map(
    //   t => (hashingTF.transform(t.sliding(3).toSeq))
    // ).cache()
    // vectors.count()

    val vectors = rawTweets.map(
      t => (hashingTF.transform(t._2))
    ).cache()
    vectors.count()

    val numClusters = 3
    val numIterations = 100
    val model = KMeans.train(vectors, numClusters, numIterations)

    // val labeledTweets = rawTweets.map{ row =>
    //   (row._1, model.predict(hashingTF.transform(row._2), row._2))
    // }
    //
    // val labeledTweetsDf = SparkSession.createDataFrame(labeledTweets).toDF("tweets", "label")


  }
}
