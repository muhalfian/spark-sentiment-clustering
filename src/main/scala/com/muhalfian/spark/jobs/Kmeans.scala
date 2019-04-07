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

object Kmeans extends StreamUtils {

  def main(args: Array[String]): Unit = {

    val SparkSession = getSparkSession(args)
    import SparkSession.implicits._


    val tweetsDF = SparkSession.read.json("/home/blade1/Documents/spark-sentiment-clustering/db/resentiment_agg.json")

    val tweetsRDD = tweetsDF.rdd

    val bagOfWord = tweetsRDD.map(
      row => {
        val label = row.getLong(0)
        val tweets = row.getString(2)
        (label, tweets.split(" ").toSeq)
      }
    )

    // val splits = bagOfWord.randomSplit(Array(0.8, 0.2), seed = 11L)
    // val training = splits(0).cache()
    // val test = splits(1).cache()

    val hashingTF = new HashingTF(2000)

    val vectors = bagOfWord.map(
      t => (hashingTF.transform(t.sliding(3).toSeq))
    ).cache()
    vectors.count()

    val model = KMeans.train(vectors, numClusters, numIterations)

    val labeledTweets = finalTweets.map{ case(tweet) =>
      (tweet, model.predict(hashingTF.transform(tweet.sliding(3).toSeq)))
    }

    val labeledTweetsDf = SparkSession.createDataFrame(labeledTweets).toDF("tweets", "label")


  }
}
