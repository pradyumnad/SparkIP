package edu.umkc.ic

/**
 * Created by pradyumnad on 10/07/15.
 */

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vector}
import org.apache.spark.{SparkConf, SparkContext}
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_features2d._
import org.bytedeco.javacpp.opencv_highgui._
import org.bytedeco.javacpp.opencv_nonfree.{SURF}

import scala.collection.mutable

object IPApp {
  val INPUT_DIR = "files/Train"

  val detector = new SURF

  val mask = new Mat
  val featureVectorsCluster = new mutable.MutableList[String]

  def surfDescriptors(fileName: String): Mat = {
    println(fileName)
    val img = imread(fileName)

    if (img.empty()) {
      println("Image is empty")
    }
    //-- Step 1: Detect the keypoints using ORB

    val keypoints = new KeyPoint
    val descriptors = new Mat

    detector.detectAndCompute(img, mask, keypoints, descriptors)

    println(s"Key Descriptors ${descriptors.rows()} x ${descriptors.cols()} ${descriptors.channels()}")

    descriptors
  }

  def bowDescriptors(fileName: String): Mat = {
    println(fileName)
    val img = imread(fileName)

    if (img.empty()) {
      println("Image is empty")
    }
    //-- Step 1: Detect the keypoints using ORB

    val extractor = new OpponentColorDescriptorExtractor
    val matcher = new BFMatcher

    val detector = new BOWImgDescriptorExtractor(extractor, matcher)

    val keypoints = new KeyPoint
    val descriptors = new Mat

    detector.compute(img, keypoints, descriptors)

    println(s"Key Descriptors ${descriptors.rows()} x ${descriptors.cols()} ${descriptors.channels()}")

    descriptors
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
        val desc = surfDescriptors(name.split(":")(1))
        //        unClusteredFeatures.push_back(desc) //Storing all the features form the Training set.

        val list = ImageUtils.matToString(desc)
        println("-- " + list.size)
        list
      }
    }.reduce((x, y) => x ::: y)

    val featuresSeq = sc.parallelize(data)

    featuresSeq.saveAsTextFile("bowfeatures")
    println("Total size : " + data.size)
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
