import java.nio.file.{Paths, Files}

import org.apache.spark.mllib.clustering.{KMeansModel, KMeans}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by pradyumnad on 12/07/15.
 */
object KMeansApp {
  def train(sc: SparkContext): Unit = {
    // Load and parse the data
    val data = sc.textFile("features2")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    clusters.save(sc, "myModelPath")
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName(s"SpamApp").setMaster("local[*]")
    val sc = new SparkContext(conf)

    if (!Files.exists(Paths.get("myModelPath"))) {
      train(sc)
    }

    val sameModel = KMeansModel.load(sc, "myModelPath")

    println(s"No of clusters : ${sameModel.k}")
    println(sameModel.clusterCenters.mkString(" "))
    val res = sameModel.predict(Vectors.dense(2, 3.5, 1.5, 1.0))
    println(s"Prediction : $res")
  }
}
