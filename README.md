# SparkIP

This project is about creating a Image classification workflow in `Apache Spark`, `JavaCV` (OpenCV).

###Workflow
___________
Listed are the steps below to achieve Image Classification

* Training Set
* Key Descriptors generation using SIFT/SURF
* K Means algorithms to "k" clusters (Called as Vocabulary)
* Use Vocabulary to generate Histograms for each image from the Training Set
* Label the Histograms
* Use Naive Bayes to generate Model
* Choose a test image and generate the Histogram image and predict it using the Naive Bayes model generated above.

###Data Set
------------

CMU Face Images `https://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/faces.data.html`
