package org.simple.stuff

import breeze.linalg.DenseVector
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, Row, SparkSession}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.optim._
import org.apache.spark.rdd.RDD

object TestRef{
  def main(args: Array[String])  : Unit = {
    print("hello")

    val conf = Engine
      .createSparkConf()
      .setAppName("bigdl-blues")
      .setMaster("local[1]")
    val ss = SparkSession.builder().config(conf).getOrCreate()
    import ss.implicits._


    Engine.init

    val csv = ss
      .read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("creditcard.csv")
//      .limit(10000)
      .cache()

//    csv.printSchema()

    val colsToRemove = Seq("V28", "V27", "V26", "V25", "V24", "V23", "V22", "V20", "V15", "V13", "V8")
    val leanCSV = csv.select(csv.columns.filter(colName => !colsToRemove.contains(colName)).map(colName => new Column(colName)):_*)

    val enrichedCSV = leanCSV
      .withColumn("V1_", when(col("V1") < -3, 1).otherwise(0))
      .withColumn("V2_", when(col("V2") > 2.5, 1).otherwise(0))
      .withColumn("V3_", when(col("V3") < -4, 1).otherwise(0))
      .withColumn("V4_", when(col("V4") > 2.5, 1).otherwise(0))
      .withColumn("V5_", when(col("V5") < -4.5, 1).otherwise(0))
      .withColumn("V6_", when(col("V6") < -2.5, 1).otherwise(0))
      .withColumn("V7_", when(col("V7") < -3, 1).otherwise(0))
      .withColumn("V9_", when(col("V9") < -2, 1).otherwise(0))
      .withColumn("V10_", when(col("V10") < -2.5, 1).otherwise(0))
      .withColumn("V11_", when(col("V11") > 2, 1).otherwise(0))
      .withColumn("V12_", when(col("V12") < -2, 1).otherwise(0))
      .withColumn("V14_", when(col("V14") < -2.5, 1).otherwise(0))
      .withColumn("V16_", when(col("V16") < -2, 1).otherwise(0))
      .withColumn("V17_", when(col("V17") < -2, 1).otherwise(0))
      .withColumn("V18_", when(col("V18") < -2, 1).otherwise(0))
      .withColumn("V19_", when(col("V19") > 1.5 -3, 1).otherwise(0))
      .withColumn("V21_", when(col("V21") > 0.6, 1).otherwise(0))
      .withColumn("Class", when(col("Class") === 0, 1).otherwise(2))
      .withColumnRenamed("Class", "Fraud")

    val fraud = enrichedCSV.filter("Fraud == 1").count()
    val normal = enrichedCSV.filter("Fraud == 0").count()

    println(s"Fraud count: $fraud")
    println(s"Normal count: $normal")

    var Array(xTrain, xTest) = enrichedCSV.randomSplit(Array(0.8, 0.2))

    val labels = Array("Fraud")
    val features = xTrain.columns.filterNot(labels.contains(_))


    for(x <- features)
    {
      if(x != "Fraud" && x != "Normal") {
        val meanFeat = enrichedCSV.select(avg(x)).first().get(0)
        val stdFeat = enrichedCSV.select(stddev(x)).first().get(0)

        xTrain = xTrain.withColumn(x, (col(x) - meanFeat) / stdFeat)
        xTest = xTest.withColumn(x, (col(x) - meanFeat) / stdFeat)
      }
    }

    val xTrainFraud = xTrain.filter(row => row.getAs[Int]("Fraud") == 1)
    List(1 to 100).foreach(_ => xTrain = xTrain.union(xTrainFraud))
    xTrain = xTrain.orderBy(rand)
    xTest = xTest.orderBy(rand)

    val split = (xTest.count() / 2).toInt
    val inputXTest = xTest.limit(split)
    val inputXValid = xTest.except(inputXTest)

    val samples : RDD[Sample[Float]]  = xTrain.rdd
      .mapPartitions((rows: Iterator[Row]) => {
        rows.map(row => {
          val feats = Tensor(DenseVector(features.map(c => row.getAs[Double](c).toFloat)))
          val labs = Tensor(DenseVector(labels.map(c => row.getAs[Int](c).toFloat)))
          Sample(feats, labs)
        })
    })

    val testRDD = inputXTest.rdd
      .mapPartitions((rows: Iterator[Row]) => {
        rows.map(row => {
          val feats = Tensor(DenseVector(features.map(c => row.getAs[Double](c).toFloat)))
          val labs = Tensor(DenseVector(labels.map(c => row.getAs[Int](c).toFloat)))
          Sample(feats, labs)
        })
      })

    val valRDD = inputXValid.rdd
      .mapPartitions((rows: Iterator[Row]) => {
        rows.map(row => {
          val feats = Tensor(DenseVector(features.map(c => row.getAs[Double](c).toFloat)))
          val labs = Tensor(DenseVector(labels.map(c => row.getAs[Int](c).toFloat)))
          Sample(feats, labs)
        })
      })


    val input_nodes = 36

////  Multiplier maintains a fixed ratio of nodes between each layer.
    val multiplier = 1.5

////  Number of nodes in each hidden layer
    val hidden_nodes1 = 72
    val hidden_nodes2 = (hidden_nodes1 * multiplier).toInt
    val hidden_nodes3 = (hidden_nodes2 * multiplier).toInt
    val hidden_nodes4 = (hidden_nodes3 * multiplier).toInt

    implicit val ev = TensorNumeric.NumericFloat
    val model = Sequential[Float]()
    model
      .add(Linear(input_nodes,hidden_nodes1))
      .add(Linear(hidden_nodes1, hidden_nodes2))
      .add(Linear(hidden_nodes2, hidden_nodes3))
      .add(Linear(hidden_nodes3, hidden_nodes4))
      .add(GaussianDropout(0.5))
      .add(Linear(hidden_nodes4, 2))
      .add(SoftMax())

    val training_epochs = 10
    val training_dropout = 0.9
    val display_step = 1
    val batch_size = 2048
    val learning_rate = 0.005

    val optimizer = Optimizer[Float](
      model=model,
      sampleRDD=samples,
      criterion= MSECriterion[Float](),
      batchSize = batch_size, featurePaddingParam = null, labelPaddingParam = null
    )

    val trainedModel = optimizer
      .setValidation(
        trigger = Trigger.everyEpoch,
        sampleRDD=valRDD,
        vMethods = Array(new Top1Accuracy),
        batchSize=batch_size)
      .setOptimMethod(new Adagrad(learningRate=learning_rate, learningRateDecay=0.0002))
      .setEndWhen(Trigger.maxEpoch(training_epochs))
      .optimize()

    val f = samples.filter(x => x.label() == Tensor(T(2f)))
    val nf = samples.filter(x => x.label() == Tensor(T(1f)))

    val targetedFraud = trainedModel.predictClass(ss.sparkContext.parallelize(f.collect()))
    val errorWithinFraud = (targetedFraud.filter(x => x == 1).count() / f.count().toDouble)*100

    val targetedNormal = trainedModel.predictClass(ss.sparkContext.parallelize(nf.collect()))
    val errorWithinNormal = (targetedNormal.filter(x => x == 2).count() / nf.count().toDouble)*100

    targetedFraud.take(10).foreach(println)
    println(s"predict fraud: $errorWithinFraud")
    targetedNormal.take(10).foreach(println)
    println(s"predict normal:$errorWithinNormal")
 }
}
