import argparse
import sys

import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"local": ["localhost:4444", "localhost:4443"]})
    tf_config_worker0 = tf.ConfigProto()
    tf_config_worker0.gpu_options.allow_growth = True # 自适应
    server = tf.train.Server(cluster, job_name="local", task_index=0,config=tf_config_worker0)

    server.start()
    server.join()

if __name__ =="__main__":
    tf.app.run(main=main)

