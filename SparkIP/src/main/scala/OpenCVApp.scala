/**
 * Created by pradyumnad on 10/07/15.
 */

import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_features2d._
import org.bytedeco.javacpp.opencv_highgui._
import org.bytedeco.javacpp.opencv_contrib._
import org.bytedeco.javacpp.opencv_imgproc._
import org.bytedeco.javacpp.opencv_nonfree.SURF

object OpenCVApp {

  def train(): Unit = {

  }

  def main(args: Array[String]) {
    val img_1 = imread("/Users/pradyumnad/KDM/SparkIP/files/101_ObjectCategories/airplanes/image_0010.jpg", CV_LOAD_IMAGE_GRAYSCALE)
    if (img_1.empty()) {
      println("Image is empty")
      -1
    }

    //-- Step 1: Detect the keypoints using ORB
    val detector = new SURF
    val keypoints_1 = new KeyPoint

    val mask = new Mat
    val descriptors_1 = new Mat

    detector.detectAndCompute(img_1, mask, keypoints_1, descriptors_1)

//    println(s"No of Keypoints ${keypoints_1.size()}")
    println(s"Key Descriptors ${descriptors_1.rows()} x ${descriptors_1.cols()}")

    val img_out = new Mat
    drawKeypoints(img_1, keypoints_1, img_out)

    imshow("Keypoints", img_out)

    //Making something up with BOW :P
    val bowTrainer = new BOWKMeansTrainer(100)
    bowTrainer.add(descriptors_1)

    val vocabulary = bowTrainer.cluster()

    println(vocabulary.asCvMat().toString)
    println(s"BOW Descriptors ${vocabulary.rows()} x ${vocabulary.cols()}")

    waitKey(0)
  }

  def matcher(): Unit = {
    val img_1 = imread("/Users/pradyumnad/KDM/SparkIP/files/101_ObjectCategories/airplanes/image_0001.jpg", CV_LOAD_IMAGE_GRAYSCALE)
    val img_2 = imread("/Users/pradyumnad/Desktop/2008_003703.jpg", CV_LOAD_IMAGE_GRAYSCALE)

    if (img_1.empty() || img_2.empty()) {
      println("Image is empty")
      -1
    }

    //-- Step 1: Detect the keypoints using ORB
    val brisk = new BRISK
    println(brisk)
    val detector = new ORB
    val keypoints_1 = new KeyPoint
    val keypoints_2 = new KeyPoint

    val mask = new Mat

    val descriptors_1 = new Mat
    val descriptors_2 = new Mat
    detector.detectAndCompute(img_1, mask, keypoints_1, descriptors_1)
    detector.detectAndCompute(img_2, mask, keypoints_2, descriptors_2)

    //-- Step 3: Matching descriptor vectors with a brute force matcher
    val matches = new DMatchVectorVector
    val bf = new BFMatcher
    bf.knnMatch(descriptors_2, descriptors_1, matches, 1)

    //-- Draw matches
    val img_matches = new Mat
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches)
    //-- Show detected matches
    imshow("Matches", img_matches)
    waitKey(0)
  }
}
