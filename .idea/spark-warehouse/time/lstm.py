# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

import numpy as np
import tensorflow as tf

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

#spark预处理数据
from pyspark.conf import SparkConf
import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
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
conf=SparkConf().setMaster("spark://lf-MS-7976:7077")
# os.environ['JAVA_HOME'] = conf.get(SECTION, 'JAVA_HOME')
spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
sc =spark.sparkContext
sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pyhdfs as pd
fs = pd.HdfsClient("127.0.0.1", 9000)
#spark 预处理数据
if(not fs.exists("/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_ok.txt")):
     sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001.txt")\
    .map(lambda x:str(x).split(",")).map(lambda x:float(x[1])).filter(lambda x:True if x<100 and x>0 else False).repartition(1).saveAsTextFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_ok.txt")

rdd=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_ok.txt",1).map(lambda x:float(x))
num=rdd.count()
rdd_num=sc.parallelize(range(num),1)
input=np.array(rdd_num.zip(rdd).map(lambda x:[x[0],x[1]]).collect())
input_train=np.array(rdd_num.zip(rdd).map(lambda x:[x[0],x[1]]).filter(lambda x:x[0]>3000 and x[0]<6000).take(2000))
print(input_train)
print(max(input_train[:,1]))
print(min(input_train[:,1]))

class _LSTMModel(ts_model.SequentialTimeSeriesModel):
        """A time series model-building example using an RNNCell."""

        def __init__(self, num_units, num_features, dtype=tf.float32):
            """Initialize/configure the model object.
            Note that we do not start graph building here. Rather, this object is a
            configurable factory for TensorFlow graphs which are run by an Estimator.
            Args:
              num_units: The number of units in the model's LSTMCell.
              num_features: The dimensionality of the time series (features per
                timestep).
              dtype: The floating point data type to use.
            """
            super(_LSTMModel, self).__init__(
                # Pre-register the metrics we'll be outputting (just a mean here).
                train_output_names=["mean"],
                predict_output_names=["mean"],
                num_features=num_features,
                dtype=dtype)
            self._num_units = num_units
            # Filled in by initialize_graph()
            self._lstm_cell = None
            self._lstm_cell_run = None
            self._predict_from_lstm_output = None

        def initialize_graph(self, input_statistics):
            """Save templates for components, which can then be used repeatedly.
            This method is called every time a new graph is created. It's safe to start
            adding ops to the current default graph here, but the graph should be
            constructed from scratch.
            Args:
              input_statistics: A math_utils.InputStatistics object.
            """
            super(_LSTMModel, self).initialize_graph(input_statistics=input_statistics)
            self._lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._num_units)
            # Create templates so we don't have to worry about variable reuse.
            self._lstm_cell_run = tf.make_template(
                name_="lstm_cell",
                func_=self._lstm_cell,
                create_scope_now_=True)
            # Transforms LSTM output into mean predictions.
            self._predict_from_lstm_output = tf.make_template(
                name_="predict_from_lstm_output",
                func_=lambda inputs: tf.layers.dense(inputs=inputs, units=self.num_features),
                create_scope_now_=True)

        def get_start_state(self):
            """Return initial state for the time series model."""
            return (
                # Keeps track of the time associated with this state for error checking.
                tf.zeros([], dtype=tf.int64),
                # The previous observation or prediction.
                tf.zeros([self.num_features], dtype=self.dtype),
                # The state of the RNNCell (batch dimension removed since this parent
                # class will broadcast).
                [tf.squeeze(state_element, axis=0)
                 for state_element
                 in self._lstm_cell.zero_state(batch_size=1, dtype=self.dtype)])

        def _transform(self, data):
            """Normalize data based on input statistics to encourage stable training."""
            mean, variance = self._input_statistics.overall_feature_moments
            return (data - mean) / variance

        def _de_transform(self, data):
            """Transform data back to the input scale."""
            mean, variance = self._input_statistics.overall_feature_moments
            return data * variance + mean

        def _filtering_step(self, current_times, current_values, state, predictions):
            """Update model state based on observations.
            Note that we don't do much here aside from computing a loss. In this case
            it's easier to update the RNN state in _prediction_step, since that covers
            running the RNN both on observations (from this method) and our own
            predictions. This distinction can be important for probabilistic models,
            where repeatedly predicting without filtering should lead to low-confidence
            predictions.
            Args:
              current_times: A [batch size] integer Tensor.
              current_values: A [batch size, self.num_features] floating point Tensor
                with new observations.
              state: The model's state tuple.
              predictions: The output of the previous `_prediction_step`.
            Returns:
              A tuple of new state and a predictions dictionary updated to include a
              loss (note that we could also return other measures of goodness of fit,
              although only "loss" will be optimized).
            """
            state_from_time, prediction, lstm_state = state
            with tf.control_dependencies(
                    [tf.assert_equal(current_times, state_from_time)]):
                transformed_values = self._transform(current_values)
                # Use mean squared error across features for the loss.
                predictions["loss"] = tf.reduce_mean(
                    (prediction - transformed_values) ** 2, axis=-1)
                # Keep track of the new observation in model state. It won't be run
                # through the LSTM until the next _imputation_step.
                new_state_tuple = (current_times, transformed_values, lstm_state)
            return (new_state_tuple, predictions)

        def _prediction_step(self, current_times, state):
            """Advance the RNN state using a previous observation or prediction."""
            _, previous_observation_or_prediction, lstm_state = state
            lstm_output, new_lstm_state = self._lstm_cell_run(
                inputs=previous_observation_or_prediction, state=lstm_state)
            next_prediction = self._predict_from_lstm_output(lstm_output)
            new_state_tuple = (current_times, next_prediction, new_lstm_state)
            return new_state_tuple, {"mean": self._de_transform(next_prediction)}

        def _imputation_step(self, current_times, state):
            """Advance model state across a gap."""
            # Does not do anything special if we're jumping across a gap. More advanced
            # models, especially probabilistic ones, would want a special case that
            # depends on the gap size.
            return state

        def _exogenous_input_step(
                self, current_times, current_exogenous_regressors, state):
            """Update model state based on exogenous regressors."""
            raise NotImplementedError(
                "Exogenous inputs are not implemented for this example.")

if __name__ == '__main__':
    with tf.device("/gpu:1"):
        tf.logging.set_verbosity(tf.logging.INFO)
        # csv_file_name ="./data/multivariate_periods.csv"
        # reader = tf.contrib.timeseries.CSVReader(
        # csv_file_name,
        # column_names=((tf.contrib.timeseries.TrainEvalFeatures.TIMES,)
        #               + (tf.contrib.timeseries.TrainEvalFeatures.VALUES,)))
        data = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: input[:,0],
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: input[:,1],
        }

        reader = NumpyReader(data)

        train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
            reader, batch_size=200, window_size=100)


        estimator = ts_estimators.TimeSeriesRegressor(
            model=_LSTMModel(num_features=1, num_units=128),
            optimizer=tf.train.AdamOptimizer(0.001),model_dir="/tmp/lf")

        estimator.train(input_fn=train_input_fn, steps=6000)

        data_train = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: input_train[:,0],
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: input_train[:,1],
        }
        reader = NumpyReader(data_train)

        evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
        evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1,checkpoint_path="/tmp/lf/model.ckpt-6000")
        # Predict starting after the evaluation
        # (predictions,) = tuple(estimator.predict(
        #     input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
        #         evaluation, steps=200)))

        observed_times = evaluation["times"][0]
        observed = evaluation["observed"][0, :, :]
        evaluated_times = evaluation["times"][0]
        evaluated = evaluation["mean"][0]
        print(evaluated)
        # predicted_times = predictions['times']
        # predicted = predictions["mean"]

        plt.figure(figsize=(15, 5))
        # plt.xlim((0, 12) )
        plt.ylim((0,15))
        # plt.axvline(999, linestyle="dotted", linewidth=4, color='r')
        observed_lines = plt.plot(observed_times, observed, label="observation", color="r")
        evaluated_lines = plt.plot(evaluated_times, evaluated, label="evaluation", color="g")
        # predicted_lines = plt.plot(predicted_times, predicted, label="prediction", color="r")
        plt.legend(handles=[observed_lines[0], evaluated_lines[0]],
                            #predicted_lines[0]],
                   loc="upper left")
        plt.savefig('predict_result.jpg')