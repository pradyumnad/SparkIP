import java.nio.file.{Paths, Files}

import org.apache.spark.mllib.classification.{NaiveBayesModel, NaiveBayes}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by pradyumnad on 19/07/15.
 */
object NBApp {

  def train(sc: SparkContext): Unit = {
    val data = sc.textFile("files/sample_naive_bayes_data.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    // Split data into training (60%) and test (40%).
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    // Save and load model
    model.save(sc, "NBModelPath")
  }

  def main(args: Array[String]) {
    println("Hey NB")

    val conf = new SparkConf().setAppName(s"NBApp").setMaster("local[*]")
    val sc = new SparkContext(conf)

    if (!Files.exists(Paths.get("NBModelPath"))) {
      train(sc)
    }

    val nbModel = NaiveBayesModel.load(sc, "NBModelPath")

    println(nbModel.labels.mkString(" "))

    val testData = Vectors.dense(Array(0.0, 0.0, 1.0))
    println(nbModel.predict(testData))
  }
}
