import java.nio.ByteBuffer

import org.apache.spark.{SparkContext, SparkConf}
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_highgui._

import scala.collection.mutable

/**
 * Created by pradyumnad on 15/07/15.
 */
object POCApp {
  def poc_save(sc: SparkContext): Unit = {
    var values = new mutable.MutableList[String]

    values += "0.0, 0.0, 0.0, 0.0"

    val data = sc.parallelize(values)

    data.saveAsTextFile("poc")
    println(data)
  }

  def main(args: Array[String]) {
    println("Hey")

    val conf = new SparkConf()
      .setAppName(s"POCApp")
      .setMaster("local")
    val sc = new SparkContext(conf)

    val lines = sc.textFile("features2")

    println(lines.take(1))
    println(lines.count())

    sc.stop()
  }
}
