from hostlist import expand_hostlist  # pip install python-hostlist
import os
import tensorflow as tf
import logging
import json

task_index = int(os.environ['SLURM_PROCID'])
n_tasks = int(os.environ['SLURM_NPROCS'])
tf_hostlist = [("%s:22222" % host) for host in
               expand_hostlist(os.environ['SLURM_NODELIST'])]
cluster = tf.train.ClusterSpec({"worker": tf_hostlist})
server = tf.distribute.Server(
    cluster.as_cluster_def(), job_name="MADZIK_TRAIN", task_index=task_index)


# This is the code that will be run on each worker
# logging.log(f"Starting worker {task_index}/{n_tasks}")
