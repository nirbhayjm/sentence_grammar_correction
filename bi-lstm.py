#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import shelve
import sys,os
import time

# Progress Bar
from progressbar import ETA, Bar, Percentage, ProgressBar, FormatLabel

# Performance metrics
from sklearn.metrics import precision_recall_fscore_support

import visualize_output as VO

import logging
logging.basicConfig(level=logging.DEBUG, 
    format='%(asctime)s :: %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p')#,

tf.reset_default_graph()

# Parameters
learning_rate = 0.01
display_step = 10
POS_SIZE = 50


# Loading input
trainFile = "tmp/trainRecords.tfr"
testFile = "tmp/testRecords.tfr"
embeddingDict = shelve.open("./tmp/embedding_table")
embeddingMat = embeddingDict['table']

VOCAB_SIZE = embeddingMat.shape[0]
EMBEDDING_SIZE = embeddingMat.shape[1]

# Network Parameters
# n_steps = seq_max_len
n_input = EMBEDDING_SIZE + POS_SIZE # Word embedding size + POS one-hot vector size
n_hidden = 512 # hidden layer num of features
n_classes = 3
batch_size = 128
num_epochs = 200

# Dataset parameters
total_num_examples = 57151
test_batch_size = 2*batch_size
train_iters = int(total_num_examples*0.8/batch_size)
test_iters = int(total_num_examples*0.2/test_batch_size)

# with tf.device('/cpu:0'):
# tf Graph input
x = tf.placeholder(tf.int64, [None, None], name="x")
# p = tf.placeholder(tf.int64, [None, None], name="pos_tag")
y = tf.placeholder(tf.int64, [None, None], name="y")

# A placeholder for indicating each sequence length
seqlens = tf.placeholder(tf.int64, [None], name="seqlens")
maxLen = tf.cast(tf.reduce_max(seqlens),tf.int32)

# Define weights
weights = {
    # 'hidden' : tf.Variable(tf.random_normal([n_input, n_hidden])) 
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    # 'hidden' : tf.Variable(tf.random_normal([n_hidden])) 
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Word embedding table :: Trainable and initialized using pre-trained embeddings
ET_init = tf.Variable(tf.convert_to_tensor(np.array(embeddingMat)))

x_vec = tf.nn.embedding_lookup(ET_init, x, validate_indices=True)
# p_vec = tf.one_hot(indices=p,depth=46)
# Shape of p_vec: (batch_size, n_steps, 46)
# y_oneHot = tf.one_hot(indices=y,depth=3)

# Adapted from:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
def DynamicRNN(x, weights, biases):
    # Current data input shape: (batch_size, n_steps, n_input)
    # x = tf.transpose(x, [1, 0, 2])
    # # Reshape to (n_steps*batch_size, n_input)
    # x = tf.reshape(x, [-1, n_input])
    # # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.unpack(x,0)
    x_rev = tf.reverse_sequence(x,seqlens,seq_dim=1,batch_dim=0)

    # def rnn_outputs(input,reuse):
    #     with tf.variable_scope("rnn_outputs",reuse=reuse):
    #         cell_fw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,
    #                                     state_is_tuple=True)
    #         outputs_fw,_ = tf.nn.dynamic_rnn(cell_fw, x, 
    #                             dtype=tf.float32,sequence_length=seqlens)
    #         return outputs_fw

    # with tf.variable_scope('dyn_rnn') as scope:
    with tf.variable_scope('forward'):
        # Forward direction cell
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,
                                    state_is_tuple=True)
        outputs_fw,forward_end_state = tf.nn.dynamic_rnn(cell_fw, x, 
                            dtype=tf.float32,sequence_length=seqlens)

    with tf.variable_scope('backward'):
        # Backward direction cell
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,
                                    state_is_tuple=True)
    # with tf.variable_scope('backward'):
    # with tf.variable_scope('rnn_outputs') as scope:     
    # scope.reuse_variables()
        outputs_bw,_ = tf.nn.dynamic_rnn(cell_bw, x_rev,
                            dtype=tf.float32,
                            initial_state=forward_end_state,
                            sequence_length=seqlens)

    # print "outputs_fw:",outputs_fw
    # outputs_fw = rnn_outputs(x,False)
    # outputs_bw = rnn_outputs(x_rev,True)
    b_size = tf.shape(x)[0]

    outputs = tf.concat(2,[outputs_fw,outputs_bw])
    # print "outputs shape:",outputs

    # tiled_weights = tf.tile(tf.expand_dims(weights['out'],0),[b_size,1,1])
    # print "Shape of tiled weights:",tiled_weights

    outputs_comp = tf.reshape(outputs,[-1,2*n_hidden])
    # print "outputs_comp shape:",outputs_comp
    
    # print "maxLen:",maxLen
    intermediate_mat = tf.matmul(outputs_comp,weights['out']) + biases['out']
    # intermediate_mat = tf.batch_matmul(outputs_comp,tiled_weights) #+ biases['out']
    pred = tf.reshape( intermediate_mat ,[b_size,maxLen,n_classes])

    # Pred has shape (batch_size, n_steps, n_classes)
    return pred

# rnn_input = tf.concat(2,[x_vec,p_vec])
rnn_input = x_vec
pred = DynamicRNN(rnn_input, weights, biases)
# Pred has shape (batch_size, n_steps, n_classes)
logits = tf.reshape(pred,[-1,n_classes])
# Logits : [ batch_size*n_steps, n_classes ]

# Initial y shape: (batch_size, n_steps)
y_flat = tf.reshape(y,[-1])

ratio = (1.0)/(1.0+99.0)
class_weights = tf.constant([1,ratio, 1.0 - ratio])
weighted_logits = tf.mul(logits, class_weights)

# Define loss and optimizer
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                weighted_logits, y_flat)

# Mask the losses
mask = tf.sign(tf.to_float(y_flat))
masked_losses = tf.reshape(mask*losses,  tf.shape(y))
# Masked_losses now has shape (batch_size, n_steps)

# Calculate mean loss
mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / \
                                tf.to_float(seqlens)

mean_loss = tf.reduce_mean(mean_loss_by_example)

cost = tf.reduce_mean(mean_loss)

# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Defining the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Clipping the gradients
grads = optimizer.compute_gradients(cost)
clipped_grads = [(tf.clip_by_value(grad, -15., 15.), var) for grad, var in grads]
optimizer_update = optimizer.apply_gradients(clipped_grads)

prediction_output = tf.argmax(pred,2)

# Batch accuracy
# correct_pred = tf.equal(y,prediction_output)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Op to save and restore variables
saver = tf.train.Saver()

def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename],
                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        # "pos_tags": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    x = sequence_parsed['tokens']
    y = sequence_parsed['labels']
    # p = sequence_parsed['pos_tags']
    l = context_parsed['length']

    return x,y,l #,p

def confusionScore(y_pred,y_true,lengths):
    y_pred_all = []
    y_true_all = []
    y_pred = y_pred.astype(np.int32)

    for i in range(len(lengths)):
        # print "\nSample:",y_pred[i,:lengths[i]]
        y_pred_all += list(y_pred[i,:lengths[i]])
        y_true_all += list(y_true[i,:lengths[i]])
    print zip(y_true_all,y_pred_all)
    return precision_recall_fscore_support(y_true_all,y_pred_all,
                labels=[1,2],pos_label=2)

# Batch fetching
with tf.device('/cpu:0'):
    eX,eY,eL = read_and_decode_single_example(trainFile)

    batch_x,batch_y,batch_l = tf.train.batch(
        tensors = [eX,eY,eL],
        batch_size = batch_size,
        dynamic_pad = True,
        name = 'batches'
    )

with tf.device('/cpu:0'):
    tX,tY,tL = read_and_decode_single_example(testFile)

    test_x,test_y,test_l = tf.train.batch(
        tensors = [tX,tY,tL],
        batch_size = test_batch_size,
        dynamic_pad = True,
        name = 'batches'
    )

# Launch the graph
with tf.Session() as sess:
    logging.info("Main begins!")

    save_root = "./tmp/tf_model_saves/"
    save_model_name = "grammar-BiLSTM_"+\
        "_nhidden-"+str(n_hidden)+\
        "_ne-"+str(num_epochs)+\
        "_bs-"+str(batch_size)+\
        ".ckpt"
    logging.info("Save model name: "+save_model_name)
    
    sess.run(init)

    if len(sys.argv) > 1:
        restore_vars = bool(int(sys.argv[1]))
    else:
        restore_vars = False

    logging.info("Initializing queue runners")
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    logging.info("Initializing threads")
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    if not restore_vars:
        logging.info("Training iters:"+str(train_iters))
        try:
            logging.info("Beginning training phase...")
            for epoch in range(1,num_epochs+1):
            # while coord.should_stop():
                if coord.should_stop():
                    print "Breaking due to cood.should_stop()"
                    break
                training_loss = 0.0

                widgets = ["Epoch [%2d/%2d]|" % (epoch,num_epochs),\
                            FormatLabel(''), Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=train_iters, widgets=widgets)
                pbar.start()

                # while step * FLAGS.batch_size < train_iters:
                start_time = time.time()
                for i in range(train_iters):

                    bX,bY,bL = sess.run([batch_x,batch_y,batch_l])

                    feed_dict={x: bX, y: bY, seqlens: bL}

                    # Run optimization op (backprop)
                    sess.run(optimizer_update, feed_dict=feed_dict)

                    if i%display_step == 0 and i != 0:
                        duration = (time.time() - start_time)/display_step
                        start_time = time.time()
                        CostValue = sess.run(cost, feed_dict=feed_dict)
                        # widgets[1] = FormatLabel(' Pr:%.2f Re:%.2f, F:%.2f |'\
                        #     % (Pr[0],Re[0],F[0]) )
                        widgets[1] = FormatLabel('Cost:%f | Minibatch time: %fs |' % (CostValue,duration) )
                        
                    pbar.update(i)
                if epoch%10 == 0:
                    save_path = saver.save(sess,save_root+save_model_name)
                    save_path = saver.save(sess,save_root+save_model_name+"-epoch-stage-"+str(epoch))
                    logging.info("\n\nModel saved in file: %s" % save_path)

        except tf.errors.OutOfRangeError:
            # Save variables to a file
            save_path = saver.save(sess,save_root+save_model_name)
            logging.info("\n\nModel saved in file: %s" % save_path)
        finally:
            # When done, ask the threads to stop.
            save_path = saver.save(sess,save_root+save_model_name)
            logging.info("\n\nModel saved in file: %s" % save_path)
            coord.request_stop()
    else:
        if os.path.isfile(save_root+save_model_name):
            saver.restore(sess,save_root+save_model_name)
            logging.info("Model Restored!")
        else:
            assert False, "Model not found!"
        try:
            logging.info("Beginning testing phase...")
            f = open("OutputScores.txt","a")
            # prediction_dump = open("Predictions.txt","w")
            f.write("Beginning\n")
            logging.info("Training iters:"+str(test_iters))
            x_list = []
            y_pred_list = []
            y_true_list = []
            len_list = []

            widgets = ["Test Iters:" ,\
                        FormatLabel(''), Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=test_iters, widgets=widgets)
            pbar.start()

            for i in range(1,test_iters+1):
            # for i in range(1,5+1):
                bX,bY,bL = sess.run([test_x,test_y,test_l])
                feed_dict={x: bX, seqlens: bL}
                y_predicted = sess.run(prediction_output, feed_dict=feed_dict)

                y_pred_list.append(
                    np.pad(y_predicted,((0,0),(0,212-y_predicted.shape[1])),'constant')
                )
                y_true_list.append(
                    np.pad(bY,((0,0),(0,212-bY.shape[1])),'constant')
                )
                x_list.append(
                    np.pad(bX,((0,0),(0,212-bX.shape[1])),'constant')
                )
                len_list.append(bL)

                pbar.update(i)

            logging.info("Accumulating predictions...")
            print "Shape of y_pred_list item:",y_pred_list[0].shape
            y_predicted = np.concatenate(y_pred_list,0)
            y_true = np.concatenate(y_true_list,0)
            x_all = np.concatenate(x_list,0)
            lengths = np.concatenate(len_list,0)
            Pr,Re,Fs,_ = confusionScore(y_predicted,y_true,lengths)
            pred_string = "Pr1 : %f, Re1: %f, F1: %f\n" % (Pr[0],Re[0],Fs[0])
            pred_string += "Pr2 : %f, Re2: %f, F2: %f\n" % (Pr[1],Re[1],Fs[1])
                
            VO.savePredictions(x_all,y_predicted,y_true,lengths)

            # for y_t,y_p in zip(y_true,y_predicted):
            #     prediction_dump.write(" ".join)
            # prediction_dump.close()
            print "\n\nPrediction outputs:"+pred_string
            f.write(pred_string)
            f.close()
        except tf.errors.OutOfRangeError:
            # Save variables to a file
            save_path = saver.save(sess,"./tmp/"+save_model_name)
            logging.info("\n\nModel saved in file: %s" % save_path)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    logging.info("\nOptimization Finished!")