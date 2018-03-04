import tensorflow as tf
import cifar10_input
import os
import numpy as np
import re
import sys
import tarfile

from six.moves import urllib

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
    'batch_size', 128, """Number of images to process in a batch""")
tf.app.flags.DEFINE_string(
    'data_dir', '../cifar10_data', """Path to the CIFAR-10 data directory.""")

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 300.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.0001
conv_weight_decay = 0.00005

TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(
        stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(
        data_dir=data_dir, batch_size=FLAGS.batch_size)
    return images, labels


def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(
        eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size)
    return images, labels


def inference(images, keep_prob):
        # images size is [batch,32,32,3]
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[5, 5, 3, 192], stddev=5e-2, wd=conv_weight_decay)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

        # conv1 size is [batch,32,32,192]
    with tf.variable_scope('cccp1') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[1, 1, 192, 160], stddev=5e-2, wd=conv_weight_decay)
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu(
            'biases', [160], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        cccp1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(cccp1)

        # cccp1 size is [batch,32,32,160]
    with tf.variable_scope('cccp2') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[1, 1, 160, 96], stddev=5e-2, wd=conv_weight_decay)
        conv = tf.nn.conv2d(cccp1, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        cccp2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(cccp2)

        # cccp2 size is [batch,32,32,96]
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(cccp2, ksize=[1, 3, 3, 1], strides=[
                               1, 2, 2, 1], padding="SAME", name='pool1')

        # pool1 size is [batch,16,16,96]
    with tf.variable_scope('dropout1') as scope:
        dropout = tf.nn.dropout(pool1, keep_prob, name='dropout1')

        # dropout size is [batch,16,16,96]
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[5, 5, 96, 192], stddev=5e-2, wd=conv_weight_decay)
        conv = tf.nn.conv2d(dropout, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

        # conv2 size is [batch,16,16,192]
    with tf.variable_scope('cccp3') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[1, 1, 192, 192], stddev=5e-2, wd=conv_weight_decay)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        cccp3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(cccp3)

        # cccp3 size is [batch,16,16,192]
    with tf.variable_scope('cccp4') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[1, 1, 192, 192], stddev=5e-2, wd=conv_weight_decay)
        conv = tf.nn.conv2d(cccp3, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        cccp4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(cccp4)

        # cccp4 size is [batch,16,16,192]
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(cccp4, ksize=[1, 3, 3, 1], strides=[
                               1, 2, 2, 1], padding="SAME", name='pool2')

        # pool2 size is [batch,8,8,192]
    with tf.variable_scope('dropout2') as scope:
        dropout2 = tf.nn.dropout(pool2, keep_prob, name='dropout2')
        # dropout2 size is [batch,8,8,192]
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[3, 3, 192, 192], stddev=5e-2, wd=conv_weight_decay)
        conv = tf.nn.conv2d(dropout2, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3)

        # conv3 size is [batch,8,8,192]

    with tf.variable_scope('cccp5') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[1, 1, 192, 192], stddev=5e-2, wd=conv_weight_decay)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        cccp5 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(cccp5)

        # cccp5 size is [batch,8,8,192]
    with tf.variable_scope('cccp6') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[1, 1, 192, 10], stddev=5e-2, wd=conv_weight_decay)
        conv = tf.nn.conv2d(cccp5, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        cccp6 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(cccp6)

        # cccp6 size is [batch,8,8,10]
    logits = tf.reduce_mean(cccp6, [1, 2])
    return logits


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


# if __name__ == "__main__":
#     a = np.ones([10, 32, 32, 3], dtype=np.float32)
#     c = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 9], dtype=tf.int32)
#     a_ = tf.constant(a)
#     b = inference(a_)
#     d = tf.nn.in_top_k(b, c, 1)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print(sess.run(b))

#         print(np.sum(sess.run(d)))
