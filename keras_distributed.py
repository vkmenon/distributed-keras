'''
Adapted from: https://gist.github.com/fchollet/2c9b029f505d94e6b8cd7f8a5e244a4e (@fchollet)
-this version works on Keras 2.0.x and Tensorflow 1.3.x and 1.4.x 

Script to train a Keras model in a distributed (multi-node) environment.
This script adpots the Asynchronous Data-Parallel training paradigm.

Given a model built with Keras or tf.keras and saved as a .json file, the script will 
distribute the training of that model.

Script must be run on each node (and each node must have all the required packages installed).

Example Usage:

	On Node 1 (Parameter Server): 
    $ python keras_distributed.py --ps_hosts="hostname1:port" --worker_hosts="hostname1:port,hostname2:port" --job_name="ps" --task_index=0

    On Node 1 (Worker):
    $ python keras_distributed.py --ps_hosts="hostname1:port" --worker_hosts="hostname1:port,hostname2:port" --job_name="worker" --task_index=0

    On Node 2 (Worker)
    $ python keras_distributed.py --ps_hosts="hostname1:port" --worker_hosts="hostname1:port,hostname2:port" --job_name="worker" --task_index=1

'''

import tensorflow as tf
import numpy as np
import keras #or import tf.keras or tf.contrib.keras.api.keras
from keras.models import model_from_json
# from tf.keras.models import model_from_json - an option if keras is not installed (TF1.4+)
# from from tensorflow.contrib.keras.api.keras.models import model_from_json (TF1.2+)

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS
STEPS = 1000000

# Function to load data
def get_train_batch(batch_size,i):
    
    # implement random selection of data 

    # return appropriate numbers of inputs and corresponding labels
    data
    labels

    return (data,labels)

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    # This cluster specification tells each node in the cluster which other nodes are workers / PSs
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    
    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    # Simply join if running from a parameter server
    if FLAGS.job_name == "ps":
        server.join()
    # Train if under worker 
    # Cannot use model.fit()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
            # set Keras learning phase to train
            keras.backend.set_learning_phase(1)
            # do not initialize variables on the fly
            keras.backend.manual_variable_initialization(True)

            # load the Keras model 
            json_file = open('PATH/TO/MODEL/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)

            # Tensor that holds the keras model predictions
            preds = model.output

            # placeholder for training targets (labels)
            # replace num_classes = total number of classes
            targets = tf.placeholder(tf.float32, shape=(None, num_classes)) 

            # crossentropy loss
            xent_loss = tf.reduce_mean(
                keras.losses.categorical_crossentropy(targets, preds))

            # we create a global_step tensor for distributed training
            # (a counter of iterations)
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # apply regularizers if any
            if model.losses:
                total_loss = xent_loss * 1.  # copy tensor
                for reg_loss in model.losses:
                    total_loss = total_loss + reg_loss
            else:
                total_loss = xent_loss

            # set up TF optimizer
            optimizer = tf.train.AdamOptimizer(5e-06)

            # Set up model update ops (batch norm ops).
            # The gradients should only be computed after updating the moving average
            # of the batch normalization parameters, in order to prevent a data race
            # between the parameter updates and moving average computations.
            with tf.control_dependencies(model.updates):
                barrier = tf.no_op(name='update_barrier')

            # define gradient updates
            with tf.control_dependencies([barrier]):
                grads = optimizer.compute_gradients(
                    total_loss,
                    model.trainable_weights)
                grad_updates = optimizer.apply_gradients(grads)

            # define training operation
            from tensorflow.python.ops.control_flow_ops import with_dependencies
            train_op = with_dependencies([grad_updates],
                                                total_loss,
                                                name='train')

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            # Create a "supervisor", which oversees the training process.
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                    logdir="/tmp/train_logs",
                                    init_op=init_op,
                                    summary_op=summary_op,
                                    saver=saver,
                                    global_step=global_step,
                                    save_model_secs=600)
            # The supervisor takes care of session initialization, restoring from
            # a checkpoint, and closing when done or an error occurs.

            with sv.managed_session(server.target) as sess:
                # Loop until the supervisor shuts down or 1000000 steps have completed.
                step = 0
                count = 0
                print("session started")
                while not sv.should_stop() and step < STEPS:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.

                # feed_dict must contain the model inputs (the tensors listed in model.inputs)
                # and the "targets" placeholder we created ealier
                # it's a dictionary mapping tensors to batches of Numpy data
                # like: feed_dict={model.inputs[0]: np_train_data_batch, targets: np_train_labels_batch}
                    data, labels = get_train_batch(32,count)
                    feed_dict={model.inputs[0]: data, targets: labels}
                    loss_value, step_value = sess.run([train_op, global_step], feed_dict=feed_dict)
                    step += 1
                    print("Step:%d, Loss:%.3f" % (step,loss_value))
                    
            # Ask for all the services to stop.
            sv.stop()

if __name__ == "__main__":
  tf.app.run()
