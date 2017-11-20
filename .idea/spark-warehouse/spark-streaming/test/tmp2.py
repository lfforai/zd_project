from pyspark.conf import SparkConf
import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
import struct
from datetime import datetime

from tensorflowonspark import TFCluster
import pyspark.sql as sql_n       #spark.sql
from pyspark import SparkContext  # pyspark.SparkContext dd
from pyspark.conf import SparkConf #conf

from pyspark.sql.types import *

schema = StructType([
    StructField("id",  StringType(), True),
    StructField("value", FloatType(), True),
    StructField("date", StringType(), True)]
)

os.environ['JAVA_HOME'] = "/tool_lf/java/jdk1.8.0_144/bin/java"
os.environ["PYSPARK_PYTHON"] = "/root/anaconda3/bin/python"
os.environ["HADOOP_USER_NAME"] = "root"
conf=SparkConf().setMaster("spark://sjfx4:7077")
sc=SparkContext(conf=conf)
print(sc.textFile("hdfs://sjfx1:9000/zd_data11.14/FQ/G_CFMY_1_001FQ001.txt").take(100))
import os
status = os.system('/tool_lf/spark/spark-2.2.0-bin-hadoop2.7/bin/spark-submit'+
                   " --py-files sample_model.py,AR_model_mapfunc.py,KDE_model_mapfunc.py"+
                   " --conf spark.executorEnv.LD_LIBRARY_PATH='${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64'"+
                   " --conf spark.executorEnv.CLASSPATH='$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}'"+
                                                                                                                                                "--conf spark.executorEnv.HADOOP_HDFS_HOME='/tool_lf/hadoop/hadoop-2.7.4'"+"model_run_AR.py")
