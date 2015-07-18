package edu.umkc.ic

import org.apache.spark.mllib.linalg.{DenseVector, Matrices, Matrix, Vector}
import org.bytedeco.javacpp.opencv_core.Mat

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Created by pradyumnad on 17/07/15.
 */
object ImageUtils {

  def matToMatrix(mat: Mat): Matrix = {
    val imageCvmat = mat.asCvMat()

    val noOfCols = imageCvmat.cols()
    val noOfRows = imageCvmat.rows()

    //Channels discarded, take care of this when you are using multiple channels
    val imageInDouble = new Array[Double](noOfCols * noOfRows)

    for (row <- 0 to noOfRows - 1) {
      for (col <- 0 to noOfCols - 1) {
        val pixel = imageCvmat.get(row, col)
        imageInDouble :+ pixel
      }
    }

    println(s"Key Descriptors $noOfRows x $noOfCols")
    //    println(s"Double size ${imageInDouble.length}")

    val matrix = Matrices.dense(noOfRows, noOfCols, imageInDouble)
    matrix
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
        vecLine.append(pixel+" ")
      }

      fVectors += vecLine.toString
    }
    fVectors.toList
  }
}
