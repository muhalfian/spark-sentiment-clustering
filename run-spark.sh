hadoop fs -put /home/blade1/Documents/spark-structured-submit/target/scala-2.11/Prayuga-Streaming-assembly-0.1.jar hdfs:://blade1-node:9000/online_media/jars/Prayuga-Streaming-assembly-0.1.jar

spark-submit --class MediaStream --master spark://10.252.37.109:7077 --deploy-mode cluster --supervise  --executor-memory 5G --total-executor-cores 3 hdfs://blade1-node:9000/online_media/jars/Prayuga-Streaming-assembly-0.1.jar
