name := "RefTest"

version := "0.1"

scalaVersion := "2.11.11"

libraryDependencies ++= Seq(
  "com.intel.analytics.bigdl" % "bigdl-SPARK_2.1" % "0.4.0",
  "org.apache.spark" % "spark-core_2.11" % "2.1.1",
  "org.apache.spark" % "spark-sql_2.11" % "2.1.1"
)

