import argparse
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
from visdom import Visdom
import part_dataset
vis = Visdom()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_ae',
                    help='Model name: pointnet_ae [default: pointnet_ae]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

MAX_NUM_POINT = 1024
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


DATA_PATH = os.path.join(BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0')
TRAIN_DATASET = part_dataset.PartDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, class_choice=None, split='trainval')
TEST_DATASET = part_dataset.PartDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, class_choice=None, split='test')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, vector_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            vector = MODEL.Encoder(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            pred = MODEL.Decoder(vector, is_training_pl, bn_decay=bn_decay)
            pred_gan = MODEL.Decoder(vector_pl, is_training_pl, bn_decay=bn_decay)
            D_real = MODEL.Discriminator(pointclouds_pl, is_training_pl, bn_decay=bn_decay, reuse = False)
            D_gene = MODEL.Discriminator(pred_gan, is_training_pl, bn_decay=bn_decay, reuse = True)

            chamfer_loss = MODEL.get_loss(pointclouds_pl, pred)
            repulse_loss = MODEL.get_repulsion_loss(pred)
            loss = chamfer_loss + repulse_loss

            loss_D_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))
            loss_D_gene = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.zeros_like(D_gene))

            D_loss = loss_D_real + loss_D_gene
            G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))

            D_optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001, beta1=0.5).minimize(D_loss)
            G_optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001, beta1=0.5).minimize(G_loss)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        # Add summary writers
        # merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})
        saver.restore(sess, './log/GAN_model.ckpt')

        ops = {'pointclouds_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
               'vector_pl': vector_pl,
               'pred': pred,
               'pred_gan' : pred_gan,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'D_loss' : D_loss,
               'G_loss' : G_loss,
               'D_optimizer' : D_optimizer,
               'G_optimizer' : G_optimizer
               }

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer,epoch)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

        for epoch in range(MAX_EPOCH):
            log_string('**** GAN_EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch_GAN(sess, ops)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "GAN_model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps,seg = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
    return batch_data, batch_label

def train_one_epoch(sess, ops, train_writer,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE

    loss_sum = 0
    for batch_idx in range((int)(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, _ = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)

        # Augment batched point clouds by rotation
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val= sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)


        train_writer.add_summary(summary, step)
        loss_sum += loss_val

        if (batch_idx+1)%10 == 0:
            log_string('mean loss: %f' % (loss_sum / 10))
            loss_sum = 0
            train_writer.add_summary(summary, step)
        if (batch_idx+1)%5 == 0:
            vis.scatter(batch_data[0], Y=None, opts=dict(title="input", markersize=0.5), win="input_train")
            vis.scatter(pred_val[0], Y=None, opts=dict(title="output", markersize=0.5), win="output_train")
        print("loss_val : " + (str)(loss_val))


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    loss_sum = 0
    for batch_idx in range((int)(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, _ = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        loss_sum += loss_val

        vis.scatter(batch_data[0], Y=None, opts=dict(title="input_eval", markersize=0.5),win="input_eval")
        vis.scatter(pred_val[0], Y=None, opts=dict(title="output_eval", markersize=0.5),win="output_eval")
        print("loss_val : " + (str)(loss_val))

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))

def train_one_epoch_GAN(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE

    loss_sum = 0
    for batch_idx in range((int)(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, _ = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)

        noize = get_noize()
        # Augment batched point clouds by rotation
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['vector_pl'] : noize,
                     ops['is_training_pl']: is_training}

        _, D_loss_val = sess.run([ops['D_optimizer'], ops['D_loss']], feed_dict=feed_dict)
        _, G_loss_val, pred_val = sess.run([ops['G_optimizer'], ops['G_loss'], ops['pred_gan']], feed_dict = {ops['vector_pl'] : noize, ops['is_training_pl']: is_training})

        if (batch_idx+1)%5 == 0:
            vis.scatter(batch_data[0], Y=None, opts=dict(title="Real", markersize=0.5), win="Real")
            vis.scatter(pred_val[0], Y=None, opts=dict(title="Gene", markersize=0.5), win="Gene")
        print("D_loss : " + (str)(D_loss_val))
        print("G_loss : " + (str)(G_loss_val))

def get_noize():
    noize = np.random.normal(0,1,size=[BATCH_SIZE,1024])
    return noize


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
