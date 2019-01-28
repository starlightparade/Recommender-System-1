from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.feature import HashingTF
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import *
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark.sql.functions import *
from pyspark.sql.window import Window

conf = SparkConf()
sc = SparkContext(conf=conf)

sqlContext = SQLContext(sparkContext=sc)

k=2

path = "meta_Digital_Music.json"
df = sqlContext.read.json(path)
df.printSchema()

train = df.select(df["asin"],df["related.also_viewed"])
train2 = train.where(train["also_viewed"].isNotNull())
train2.show()
print("training size:",train2.count())

test = df.select(df["asin"],df["related.also_bought"])
test2 = test.where(test["also_bought"].isNotNull())
test2.show()


# train2.withColumn("transactions", train2["also_viewed"].append(train2["asin"]))

# def appendToArray(array, str):
# 	array.append(str)
#
# appendUdf==udf(targetArray, string)
# df = df.withColumn('transactions', appendUdf(train2["also_viewed"],train2["asin"]))
# train2.show(false)

udf_append = udf(lambda also_viewed, asin: [asin] + [x for x in also_viewed], ArrayType(StringType())) #Define UDF function

train_transactions = train2.withColumn('transactions', udf_append(train2["also_viewed"],train2["asin"]))
print("/////////////////////// after append:")
train_transactions = train_transactions.drop("also_viewed")
train.unpersist()
test.unpersist()
train2.unpersist()
# train_transactions.show(20, False)

fpGrowth = FPGrowth(itemsCol="transactions", minSupport=0.0008, minConfidence=0.0008)
model = fpGrowth.fit(train_transactions)

# Display frequent itemsets.
freq = model.freqItemsets
print("/////////////////////// frequent item set:")
# freq = freq.filter(""" size(items) >= 2 """)
freq.show()
# Display generated association rules.
assoc = model.associationRules
assoc = assoc.filter(""" size(antecedent) == 1 """).filter(""" size(consequent) == 1 """)

# assoc = assoc.orderBy("antecedent",desc("confidence"))
# topK = assoc.groupby(['antecedent'], sort=False).head(5)
window = Window.partitionBy(assoc['antecedent']).orderBy(assoc['confidence'].desc())
#
topK = assoc.select('*', rank().over(window).alias('rank')).filter(col('rank') <= k).drop(col("lift"))
print("/////////////////////// top K association rules:")

# udf_retrieve = udf(lambda a: str(a[0]), StringType())
topK = topK.withColumn("asin",concat_ws(',', col('antecedent'))).withColumn("consequent",concat_ws(',', col('consequent'))).drop(col("rank")).drop(col("confidence"))
topK.show()
topK = topK.groupBy("asin").agg(collect_set("consequent").alias("recommendation"))
topK.show()
# assoc = assoc.groupBy(col("antecedent")).agg(col("consequent"))

# # transform examines the input items against all the association rules and summarize the
# # consequents as prediction
# prediction = model.transform(train_transactions)
# # prediction2 = prediction.where(prediction["prediction"].isNotNull())
# prediction = prediction.filter(""" size(prediction) != 0 """)
# prediction.show()
# prediction2 = prediction.withColumn("prediction",array([col("prediction")[i] for i in range(k)]))
# train_transactions.unpersist()


#Define UDF function
#if at least one item in common, conversion rate = 1
udf_check = udf(lambda prediction, also_bought: 1 if [x for x in prediction if x in also_bought] else 0, IntegerType())

topK = topK.alias("topK")
test2 = test2.alias("test2")
joined_df = topK.join(test2, col("test2.asin") == col("topK.asin"), 'inner').drop(col("test2.asin"))
print("/////////////////////// join topK and also_bought:")
joined_df.show()
# compare = prediction2.select(prediction2["asin"],df["related.also_viewed"])
compare = joined_df.withColumn('conversion_rate', udf_check(joined_df["recommendation"],joined_df["also_bought"]))
print("/////////////////////// calculate conversion rate:")

compare.show()

total = compare.groupBy().sum("conversion_rate").collect()
print(total[0][0])
num_record = test2.count()
print(num_record)
print("conversion rate:",total[0][0] / num_record)

