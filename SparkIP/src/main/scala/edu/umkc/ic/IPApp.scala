package edu.umkc.ic

/**
 * Created by pradyumnad on 10/07/15.
 */

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vectors, Matrices, Matrix, Vector}
import org.apache.spark.{SparkConf, SparkContext}
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_features2d._
import org.bytedeco.javacpp.opencv_highgui._
import org.bytedeco.javacpp.opencv_nonfree.{SURF, SIFT}

import scala.collection.mutable

object IPApp {
//  val unClusteredFeatures = new Mat
  val INPUT_DIR = "files/Train"

  val detector = new SURF
  val mask = new Mat
  val featureVectorsCluster = new mutable.MutableList[String]

  def train(fileName: String): Mat = {
    println(fileName)
    val img = imread(fileName)

    if (img.empty()) {
      println("Image is empty")
      //      return Matrices.ones(1, 1)
    }
    //-- Step 1: Detect the keypoints using ORB

    val keypoints_1 = new KeyPoint
    val descriptors_1 = new Mat

    detector.detectAndCompute(img, mask, keypoints_1, descriptors_1)

    println(s"Key Descriptors ${descriptors_1.rows()} x ${descriptors_1.cols()} ${descriptors_1.channels()}")

    descriptors_1
  }

  def classify(sc: SparkContext): Unit = {

    val parsedData = sc.objectFile[Vector]("features2")

    // Cluster the data into two classes using KMeans
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    clusters.save(sc, "features_cluster")
  }

  def init(sc: SparkContext): Unit = {
    val images = sc.wholeTextFiles(s"$INPUT_DIR/*/*.jpg")
    images.map(println)

    val data = images.map {
      case (name, contents) => {
        val desc = train(name.split(":")(1))
//        unClusteredFeatures.push_back(desc) //Storing all the features form the Training set.

        val list = ImageUtils.matToString(desc)
        println("-- "+list.size)
        list
      }
    }.reduce((x, y) => x:::y)

    val featuresSeq = sc.parallelize(data)

    featuresSeq.saveAsTextFile("features2")
    println("Total size : "+data.size)
//    println("Total features : " + unClusteredFeatures.rows())
    //    println(data.take(2).toList)
  }

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName(s"IPApp")
      .setMaster("local[*]")
      .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)

    init(sc)

    //    classify(sc)

    sc.stop()
  }
}
