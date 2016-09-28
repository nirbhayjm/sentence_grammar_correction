#!/usr/bin/env python

import pickle

import tensorflow as tf

from sklearn.cross_validation import train_test_split

from progressbar import ETA, Bar, Percentage, ProgressBar, FormatLabel
import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s :: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')

trainRecordFile = "./tmp/trainRecords.tfr"
testRecordFile = "./tmp/testRecords.tfr"

def make_example(sequence,labels):
    # The object we return
    ex = tf.train.SequenceExample()
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)

    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    # fl_pos = ex.feature_lists.feature_list["pos_tags"]
    for token, label in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(label)
        # fl_pos.feature.add().int64_list.value.append(ptag)
    return ex

def writeData(tfr_file,data):
    logging.info("Writing data to TF Record file: %s" % tfr_file)
    
    widgets = [Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=len(data), widgets=widgets)
    pbar.start()

    # Write all examples into a TFRecords file
    writer = tf.python_io.TFRecordWriter(tfr_file)
    for i,(seqX,seqY) in enumerate(data):
        pbar.update(i)
        # print "SeqX:",seqX
        # print "seqY:",seqY
        # assert False
        ex = make_example(seqX, seqY)
        writer.write(ex.SerializeToString())
    writer.close()
    print("Wrote to {}".format(tfr_file))

if __name__ == '__main__':
    logging.info("Loading newData...")
    newData = pickle.load(open("./tmp/newData.txt","r"))
    logging.info("Done!")

    trainData, testData = train_test_split(newData,test_size=0.2,
                                            random_state=42)

    # print "len of trainData:",len(trainData)
    # print "len of testData:",len(testData)
    writeData(testRecordFile,testData)
    writeData(trainRecordFile,trainData)