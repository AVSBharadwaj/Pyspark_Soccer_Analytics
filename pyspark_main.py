# Databricks notebook source

from pyspark.sql.types import *

# COMMAND ----------

schema=(StructType().add("id_odsp",StringType()).add("id_event",StringType()).add("sort_order",IntegerType()).add("time",IntegerType()).add("text", StringType()).add("event_type", IntegerType()).add("event_type2", IntegerType()).add("side", IntegerType()).add("event_team", StringType()).add("opponent", StringType()).add("player", StringType()).add("player2", StringType()).add("player_in", StringType()).add("player_out", StringType()).add("shot_place", IntegerType()).add("shot_outcome", IntegerType()).add("is_goal", IntegerType()).add("location", IntegerType()).add("bodypart", IntegerType()).add("assist_method", IntegerType()).add("situation", IntegerType()).add("fast_break", IntegerType()))

# COMMAND ----------

eventsDf=(spark.read.csv("/FileStore/shared_uploads/avsbharadwajofficial@gmail.com/events.csv",schema=schema,header=True,ignoreLeadingWhiteSpace=True,ignoreTrailingWhiteSpace=True,nullValue='NA'))

# COMMAND ----------

eventsDf=eventsDf.na.fill({'player':'NA','event_team': 'NA','opponent': 'NA',
'event_type': 99, 'event_type2': 99, 'shot_place': 99,
'shot_outcome': 99, 'location': 99, 'bodypart': 99,
'assist_method': 99, 'situation': 99})

# COMMAND ----------

display(eventsDf)

# COMMAND ----------

gameDf=(spark.read.csv('/FileStore/shared_uploads/avsbharadwajofficial@gmail.com/ginf.csv',inferSchema=True,header=True,ignoreLeadingWhiteSpace=True,ignoreTrailingWhiteSpace=True,nullValue="NA"))

# COMMAND ----------

display(gameDf)

# COMMAND ----------

def mapKeyToVal(mapping):
    def mapKeyToVal_(col):
        return mapping.get(col)
    return udf(mapKeyToVal_, StringType())

# COMMAND ----------

#gameDf = gameDf.withColumn("country_code", mapKeyToVal(countryCodeMap)("country"))


# COMMAND ----------

eventsDf = (eventsDf.
withColumn("event_type_str", mapKeyToVal(evtTypeMap)("event_type")).
withColumn("event_type2_str", mapKeyToVal(evtTyp2Map)("event_type2")).
withColumn("side_str", mapKeyToVal(sideMap)("side")).
withColumn("shot_place_str", mapKeyToVal(shotPlaceMap)("shot_place")).
withColumn("shot_outcome_str", mapKeyToVal(shotOutcomeMap)("shot_outcome")).
withColumn("location_str", mapKeyToVal(locationMap)("location")).
withColumn("bodypart_str", mapKeyToVal(bodyPartMap)("bodypart")).
withColumn("assist_method_str", mapKeyToVal(assistMethodMap)("assist_method")).
withColumn("situation_str", mapKeyToVal(situationMap)("situation")))

# COMMAND ----------

joinedDf = (
eventsDf.join(gameDf, eventsDf.id_odsp == gameDf.id_odsp, "inner").
select(eventsDf.id_odsp, eventsDf.id_event, eventsDf.sort_order, eventsDf.time, eventsDf.event_type,
eventsDf.event_type_str, eventsDf.event_type2, eventsDf.event_type2_str, eventsDf.side, eventsDf.side_str, eventsDf.event_team, eventsDf.opponent, eventsDf.player, eventsDf.player2, eventsDf.player_in,
eventsDf.player_out, eventsDf.shot_place, eventsDf.shot_place_str, eventsDf.shot_outcome, eventsDf.
shot_outcome_str, eventsDf.is_goal, eventsDf.location, eventsDf.location_str, eventsDf.bodypart,
eventsDf.bodypart_str, eventsDf.assist_method, eventsDf.assist_method_str, eventsDf.situation,
eventsDf.situation_str, gameInfDf.country_code))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create Databricks/Spark database
# MAGIC CREATE DATABASE IF NOT EXISTS EURO_SOCCER_DB
# MAGIC LOCATION “dbfs:/FileStore/databricks-abhinav/eu-soccer-events/interm”
# MAGIC 
# MAGIC -- Set the database session
# MAGIC USE EURO_SOCCER_DB
# MAGIC -- Load transformed game event data into a Databricks/Spark table
# MAGIC joinedDf.write.saveAsTable(“GAME_EVENTS”, format = “parquet”, mode = “overwrite”,
# MAGIC partitionBy = “COUNTRY_CODE”, path = “dbfs:/FileStore/databricks-abhinav/eu-
# MAGIC soccer-events/interm/tr-events”)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CASE WHEN shot_place_str == ‘NA’ THEN ‘Unknown’ ELSE shot_place_str END
# MAGIC shot_place, COUNT(1) AS TOT_GOALS
# MAGIC FROM GAME_EVENTS
# MAGIC WHERE is_goal = 1
# MAGIC GROUP BY shot_place_str

# COMMAND ----------


from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

categFeatures = [“event_type_str”, “event_team”, “shot_place_str”, “location_str”,
“assist_method_str”, “situation_str”, “country_code”]

# COMMAND ----------

stringIndexers = [StringIndexer().setInputCol(baseFeature).setOutputCol(baseFeature +
“_idx”) for baseFeature in categFeatures]

encoders = [OneHotEncoder().setInputCol(baseFeature + “_idx”).setOutputCol(baseFeature
+ “_vec”) for baseFeature in categFeatures]
featureAssembler = VectorAssembler()
featureAssembler.setInputCols([baseFeature + “_vec” for baseFeature in categFeatures])
featureAssembler.setOutputCol(“features”)

# COMMAND ----------

gbtClassifier = GBTClassifier(labelCol=”is_goal”, featuresCol=”features”, maxDepth=5,
maxIter=20)

pipelineStages = stringIndexers + encoders + [featureAssembler, gbtClassifier]
pipeline = Pipeline(stages=pipelineStages)

(trainingData, testData) = gameEventsDf.randomSplit([0.75, 0.25])
model = pipeline.fit(trainingData)

# COMMAND ----------

predictions = model.transform(testData)
display(predictions.select(“prediction”, “is_goal”, “features”))
evaluator = BinaryClassificationEvaluator(
labelCol=”is_goal”, rawPredictionCol=”prediction”)
evaluator.evaluate(predictions)
