package edu.umkc.ic

import org.apache.spark.mllib.linalg.{DenseVector, Matrices, Matrix, Vector}
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_features2d.{BOWImgDescriptorExtractor, DescriptorExtractor, FlannBasedMatcher, KeyPoint}
import org.bytedeco.javacpp.opencv_highgui._
import org.bytedeco.javacpp.opencv_imgproc._
import org.bytedeco.javacpp.opencv_objdetect._
import org.bytedeco.javacpp.opencv_nonfree.{SIFT, SURF}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Created by pradyumnad on 17/07/15.
 */
object ImageUtils {

  val faceCascade: CascadeClassifier = new CascadeClassifier
  val haar_face_cascade_name = "files/haarcascade_frontalface_default.xml"
  val face_cascade_name = "files/lbpcascade_frontalface.xml"

  def init: Unit = {
    if (!faceCascade.load(face_cascade_name)) {
      println("--(!)Error loading face cascade")
    } else {
      println("Loaded face classifier")
    }
  }

  def faceMat(file: String): Mat = {
    val frame_gray: Mat = imread(file, CV_LOAD_IMAGE_GRAYSCALE)
    if (frame_gray.empty()) {
      println("Image is empty")
      -1
    }

    //Equalises the brightness and contract by normalising the histogram
    equalizeHist(frame_gray, frame_gray)

    val faces = new Rect()

    faceCascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, new Size(80, 80), new Size())

    val roiImage = frame_gray(faces)
    roiImage
  }

  def faceDetect(file: String): Mat = {
    println(file)
    val frame_gray: Mat = imread(file, CV_LOAD_IMAGE_GRAYSCALE)
    if (frame_gray.empty()) {
      println("Image is empty")
      -1
    }

    //Equalises the brightness and contract by normalising the histogram
    equalizeHist(frame_gray, frame_gray)

    val faces = new Rect()

    faceCascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0, new Size(80, 80), new Size())

    //    println(faces.size().asCvSize().toString)
    //    println("Capacity : " + faces.capacity())
    //    println(faces.asCvRect().toString)

    //     = frame_gray.adjustROI(faces.x(), faces.y(), faces.width(), faces.height())
    val roiImage = frame_gray(faces)
    // imshow("ROI", roiImage)
    println(roiImage.size().asCvSize().toString)

    val detector = new SIFT
    val keypoints_1 = new KeyPoint

    val mask = new Mat
    val descriptors = new Mat
    if (!roiImage.empty()) {
      detector.detectAndCompute(roiImage, mask, keypoints_1, descriptors)

      //    println(s"No of Keypoints ${keypoints_1.size()}")
      println(s"Key Descriptors ${descriptors.rows()} x ${descriptors.cols()}")
      descriptors
    }
    else {
      new Mat()
    }
  }

  def descriptors(file: String): Mat = {
    val img_1 = imread(file, CV_LOAD_IMAGE_GRAYSCALE)
    if (img_1.empty()) {
      println("Image is empty")
      -1
    }

    //-- Step 1: Detect the keypoints using ORB
    val detector = new SIFT(100)
    val keypoints_1 = new KeyPoint

    val mask = new Mat
    val descriptors = new Mat

    detector.detectAndCompute(img_1, mask, keypoints_1, descriptors)

    //    println(s"No of Keypoints ${keypoints_1.size()}")
    println(s"Key Descriptors ${descriptors.rows()} x ${descriptors.cols()}")
    descriptors
  }

  def bowDescriptors(file: String, dictionary: Mat): Mat = {
    val matcher = new FlannBasedMatcher()
    val detector = new SIFT()
    val extractor = DescriptorExtractor.create("SIFT")
    val bowDE = new BOWImgDescriptorExtractor(extractor, matcher)
    bowDE.setVocabulary(dictionary)
    println(bowDE.descriptorSize() + " " + bowDE.descriptorType())

//    val img = faceMat(file)
    val img = imread(file, CV_LOAD_IMAGE_GRAYSCALE)
    if (img.empty()) {
      println("Image is empty")
      -1
    }
//AN DI SA AF NE SU HA
    //5.0 2.0 6.0 0.0 3.0 1.0 4.0
    //Surprised  Sad  Happy  Angry  Afraid  Disgusted  Neutral
    val keypoints = new KeyPoint

    detector.detect(img, keypoints)

    val response_histogram = new Mat
    bowDE.compute(img, keypoints, response_histogram)

    //    println("Histogram size : " + response_histogram.size().asCvSize().toString)
    //    println("Histogram : " + response_histogram.asCvMat().toString)
    response_histogram
  }

  def matToVector(mat: Mat): Vector = {
    val imageCvmat = mat.asCvMat()

    val noOfCols = imageCvmat.cols()

    //Channels discarded, take care of this when you are using multiple channels

    val imageInDouble = new Array[Double](noOfCols)
    for (col <- 0 to noOfCols - 1) {
      val pixel = imageCvmat.get(0, col)
      imageInDouble(col) = pixel
    }
    val featureVector = new DenseVector(imageInDouble)
    featureVector
  }

  def matToVectors(mat: Mat): Array[Vector] = {
    val imageCvmat = mat.asCvMat()

    val noOfCols = imageCvmat.cols()
    val noOfRows = imageCvmat.rows()

    val fVectors = new ArrayBuffer[DenseVector]()
    //Channels discarded, take care of this when you are using multiple channels

    for (row <- 0 to noOfRows - 1) {
      val imageInDouble = new Array[Double](noOfCols)
      for (col <- 0 to noOfCols - 1) {
        val pixel = imageCvmat.get(row, col)
        imageInDouble :+ pixel
      }
      val featureVector = new DenseVector(imageInDouble)
      fVectors :+ featureVector
    }

    fVectors.toArray
  }

  def matToDoubles(mat: Mat): Array[Array[Double]] = {
    val imageCvmat = mat.asCvMat()

    val noOfCols = imageCvmat.cols()
    val noOfRows = imageCvmat.rows()

    val fVectors = new ArrayBuffer[Array[Double]]()
    //Channels discarded, take care of this when you are using multiple channels

    for (row <- 0 to noOfRows - 1) {
      val imageInDouble = new Array[Double](noOfCols)
      for (col <- 0 to noOfCols - 1) {
        val pixel = imageCvmat.get(row, col)
        imageInDouble :+ pixel
      }
      fVectors :+ imageInDouble
    }
    fVectors.toArray
  }

  def matToString(mat: Mat): List[String] = {
    val imageCvmat = mat.asCvMat()

    val noOfCols = imageCvmat.cols()
    val noOfRows = imageCvmat.rows()

    val fVectors = new mutable.MutableList[String]
    //Channels discarded, take care of this when you are using multiple channels

    for (row <- 0 to noOfRows - 1) {
      val vecLine = new StringBuffer("")
      for (col <- 0 to noOfCols - 1) {
        val pixel = imageCvmat.get(row, col)
        vecLine.append(pixel + " ")
      }

      fVectors += vecLine.toString
    }
    fVectors.toList
  }

  def vectorsToMat(centers: Array[Vector]): Mat = {

    val vocab = new Mat(centers.size, centers(0).size, CV_32FC1)

    var i = 0
    for (c <- centers) {

      var j = 0
      for (v <- c.toArray) {
        vocab.asCvMat().put(i, j, v)
        j += 1
      }
      i += 1
    }

    //    println("The Mat is")
    //    println(vocab.asCvMat().toString)

    vocab
  }

}
