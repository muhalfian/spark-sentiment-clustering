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

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}

object KmeansSVM extends StreamUtils {

  def main(args: Array[String]): Unit = {

    val SparkSession = getSparkSession(args)
    import SparkSession.implicits._

    val numClusters = 3
    val numIterations = 100
    val tweetsDF = SparkSession.read.json("/home/blade1/Documents/spark-sentiment-clustering/db/chandra_training.json")

    println(tweetsDF)

    var messages = tweetsDF.select("tweets")
    val messagesRDD = messages.rdd
    val goodBadTweets = messagesRDD.map(
      row => {
          Try{
            var processedTweet = ArrayBuffer[String]()
            //Convert tweet messages to lowercase
            val tweetMsg = row(0).toString.toLowerCase()
            //Replace @user to string 'USER_MENTION'
            var mentionlessMsg = tweetMsg.replaceAll("""@[\S]+""","USER_MENTION")
            //Replace HTTP:// to string 'LINK'
            var linklessMsg = mentionlessMsg.replaceAll("""(www\.[\S]+)|(https?:\/\/[\S]+)""","LINK")
            //Remove # from #hashtag
            var taglessMsg = linklessMsg.replaceAll("""#(\S+)""","'$1'")
            //Remove RT phrase
            var rtlessMsg = taglessMsg.replaceAll("""\brt\b""","")
            //Reduce multiple dots(.) with space
            var dotlessMsg = rtlessMsg.replaceAll("""\.{2,}"""," ")
            //Strip multiple space
            var signlessMsg = dotlessMsg.trim.replaceAll("""\s+""", " ")
            //Strip punctuation
            var punctlessMsg = signlessMsg.replaceAll("""[^\w\s\.\$]""", "")
            //Change Emoji to respectful list in emo_list (it can be either EMO_NEG or EMO_POS) reference taken from https://github.com/abdulfatir/twitter-sentiment-analysis/blob/master/preprocess.py
            var emolessMsg = punctlessMsg
            // Smile -- :), : ), :-), (:, ( :, (-:, :')
            .replaceAll("""(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))""", "EMO_POS")
            // Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
            .replaceAll("""(:\s?D|:-D|x-?D|X-?D)""", "EMO_POS")
            // Love -- <3, :*
            .replaceAll("""(<3|:\*)""", "EMO_POS")
            // Wink -- ;-), ;), ;-D, ;D, (;,  (-;
            .replaceAll("""(;-?\)|;-?D|\(-?;)""", "EMO_POS")
            // Sad -- :-(, : (, :(, ):, )-:
            .replaceAll("""(:\s?\(|:-\(|\)\s?:|\)-:)""", "EMO_NEG")
            // Cry -- :,(, :'(, :"(
            .replaceAll("""(:,\(|:\'\(|:"\()""", "EMO_NEG")

            var arrayOfWords = emolessMsg.split(" ")
            for(word <- arrayOfWords){
              if(word.matches("""^[a-zA-Z][a-z0-9A-Z\._]*$""")){
                processedTweet += word
              }
            }

            val cleanedMsg = processedTweet.mkString(" ")
            //(cleanedMsg.split(" ").toSeq)
            (cleanedMsg)
            }
          }
        )

    val exceptions = goodBadTweets.filter(_.isFailure)
    //println("Total tweets with exceptions: "+ exceptions.count())
    val finalTweets = goodBadTweets.filter(_.isSuccess).map(_.get)
    // println("Total clean tweets: "+ finalTweets.count())
    // finalTweets.take(5).foreach(x => println(x))

    val hashingTF = new HashingTF(2000)
    val vectors = finalTweets.map(
      t => (hashingTF.transform(t.sliding(3).toSeq))
    ).cache()
    vectors.count()

    val model = KMeans.train(vectors, numClusters, numIterations)

    val labeledTweets = finalTweets.map{ case(tweet) =>
      (tweet, model.predict(hashingTF.transform(tweet.sliding(3).toSeq)))
    }

    // ================= SVM ====================

    val training_labeled = labeledTweets.map(
      t => (t._2, hashingTF.transform(t._1))
    ).map(
      x => new LabeledPoint((x._1).toDouble, x._2)
    )

    println("\n\n************** Training **************\n\n")

    // Run training algorithm to build the model
    val modelSVM = new LogisticRegressionWithLBFGS()
       .setNumClasses(3)
       .run(training_labeled)

    println("\n\n************** Testing **************\n\n")

    val predictionAndLabels = test.map(
      x => {
        val prediction = modelSVM.predict(hashingTF.transform(x._2))
        (prediction, x._1.toDouble)
      }
    )

    // ====================== EVALUATION =========================

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
    // val accuracy = 1.0 * predictionAndLabels.filter(x => x._1 == x._2).count() / test.count()

    println("Training and Testing Complete, accuracy is = " + accuracy)

    // val labeledTweetsDf = SparkSession.createDataFrame(labeledTweets).toDF("tweets", "label")
    //
    // labeledTweetsDf.coalesce(2).write.format("json").save("/home/blade1/Documents/spark-sentiment-clustering/db/chandra_training_res_2.json")

  }
}
