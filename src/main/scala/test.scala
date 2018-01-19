package org.simple.stuff

import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Column, SQLContext}
import org.apache.spark.sql.functions._


object TestRef{
  def main(args: Array[String])  : Unit = {
    print("hello")

    val conf = Engine
      .createSparkConf()
      .setAppName("bigdl-blues")
      .setMaster("local[1]")
    val sc = new SparkContext(conf)
    val sqlc = new SQLContext(sc)
    Engine.init

    //  reading csv - rdd way
    //  val csv = sc.textFile("creditcard.csv")
    //  val headers = csv.first()
    //  println(s"Column names: $headers")


    val csv = sqlc
      .read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("creditcard.csv")

//    csv.printSchema()


    val colsToRemove = Seq("V28", "V27", "V26", "V25", "V24", "V23", "V22", "V20", "V15", "V13", "V8")
    val leanCSV = csv.select(csv.columns.filter(colName => !colsToRemove.contains(colName)).map(colName => new Column(colName)):_*)

    leanCSV.printSchema()

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
      .withColumn("Normal", when(col("Class") === 0, 1).otherwise(1))
      .withColumnRenamed("Class", "Fraud")

    val fraud = enrichedCSV.filter("Fraud == 1").count()
    val normal = enrichedCSV.filter("Fraud == 0").count()

    println(s"Fraud count: $fraud")
    println(s"Normal count: $normal")

    val Array(xTrainOrig, xTestOrig) = enrichedCSV.randomSplit(Array(0.8, 0.2))

    val xTrainInterim = xTrainOrig.orderBy(rand)
    val xTestInterim = xTestOrig.orderBy(rand)

    val yTrain = xTrainInterim.select("Fraud", "Normal")
    val yTest = xTestInterim.select("Fraud","Normal")

    var xTrain = xTrainInterim.select(xTrainInterim.columns.filter(x => x != "Fraud" && x != "Normal").map(colName => new Column(colName)):_*)
    var xTest = xTestInterim.select(xTestInterim.columns.filter(x => x != "Fraud" && x != "Normal").map(colName => new Column(colName)):_*)

    println("xTrain Count: " + xTrain.count() + "xTest Count: " + xTest.count() + "yTrain Count: " + yTrain.count() + "yTest Count: " + yTest.count())


//    val features = xTrain.columns
//    for(x <- features)
//    {
//      val meanFeat = mean(leanCSV.select(x))
//      val stdFeat = stddev(leanCSV.select(x))
//
//      xTrain = xTrain.withColumn(x, (col(x) - meanFeat) / stdFeat)
//      xTest = xTest.withColumn(x, (col(x) - meanFeat) / stdFeat)
//    }

//    enrichedCSV.select("V1", "V1_").where(col("V1_") === 1).take(15).foreach(println)
  }
}