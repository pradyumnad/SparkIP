import java.nio.file.{Paths, Files}

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.streaming._
import org.apache.spark.{SparkContext, SparkConf}

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

object EmailSpamApp {

  def train(sc: SparkContext, tf: HashingTF): Unit = {
    // Load 2 types of emails from text files: spam and ham (non-spam).
    // Each line has text from one email.
    val spam = sc.textFile("files/spam.txt")
    val ham = sc.textFile("files/ham.txt")

    // Create a HashingTF instance to map email text to vectors of 100 features.

    // Each email is split into words, and each word is mapped to one feature.
    val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
    val hamFeatures = ham.map(email => tf.transform(email.split(" ")))

    // Create LabeledPoint datasets for positive (spam) and negative (ham) examples.
    val positiveExamples = spamFeatures.map(features => LabeledPoint(1, features))
    val negativeExamples = hamFeatures.map(features => LabeledPoint(0, features))

    val trainingData = positiveExamples ++ negativeExamples
    trainingData.cache() // Cache data since Logistic Regression is an iterative algorithm.

    // Create a Logistic Regression learner which uses the LBFGS optimizer.
    val lrLearner = new LogisticRegressionWithSGD()
    // Run the actual learning algorithm on the training data.
    val model = lrLearner.run(trainingData)
    model.save(sc, "files/email")
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName(s"SpamApp").setMaster("local[*]")
    //    val sc = new SparkContext(conf)

    /* Streaming Context creation */
    val ssc = new StreamingContext(conf, Seconds(2))
    val sc = ssc.sparkContext
    val tf = new HashingTF(numFeatures = 100)

    if (!Files.exists(Paths.get("files/email"))) {
      train(ssc.sparkContext, tf)
    }

    val model = LogisticRegressionModel.load(sc, "files/email")

    val lines = ssc.socketTextStream("localhost", 9999)

    val cstream = lines.filter(rdd => {
      val words = rdd.split(" ")
      val example = tf.transform(words)
      val p = model.predict(example)
      println(s"Prediction : $p")
      true
    })

    cstream.print()

    ssc.start()
    ssc.awaitTermination()
  }
}
