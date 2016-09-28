#!/usr/bin/env python

import shelve
import pickle
import numpy as np

from gensim.models import Word2Vec

import logging
import os,sys

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s :: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')#,
    # filename='logfile.log')

DEBUG_PROGRESS = True
logging.info('Logger Initialized!')

vectorsPath = "/home/nirbhaym/GoogleNews-vectors-negative300.bin"
EMBEDDING_SIZE = 300

w2vDict = shelve.open("map_w2v/w2v_map_dict")
tokDict = shelve.open("map_w2v/uid_dict")
embeddingDict = shelve.open("./tmp/embedding_table")
wordVecModel = None

def wordVecInit():
    global wordVecModel
    try:
        wordVecModel = Word2Vec.load_word2vec_format(vectorsPath, binary=True)   
    except IOError:
        assert False,"Word vectors binary file could not be loaded."
    wordVecModel.init_sims(replace=True)
    wvModelExists = True

def vectorMapping(wordList):
    print "Loading w2v pre-trained vectors!"
    wordVecInit()

    vocabItemCounter = 1
    wordsInVocab = filter(lambda x: x in wordVecModel.vocab,wordList)
    wordsUnknown = filter(lambda x: x not in wordVecModel.vocab,wordList)

    print "Number of words detected in vocab:",len(wordsInVocab)
    print "Number of words not found:",len(wordList) - len(wordsInVocab)

    w2vDict['__NOT_A_WORD__'] = (0,np.zeros(300,dtype=np.float32))
    tokDict[str(0)] = '__NOT_A_WORD__'

    for word in wordsInVocab:
        w2vDict[word] = (vocabItemCounter,wordVecModel[word])
        tokDict[str(vocabItemCounter)] = word
        vocabItemCounter += 1

    for word in wordsUnknown:
        w2vDict[word] = (vocabItemCounter,np.zeros(300,dtype=np.float32))
        tokDict[str(vocabItemCounter)] = word
        vocabItemCounter += 1

def saveEmbeddingTable():
    vocabSize = 0
    for key in w2vDict:
        vocabSize += 1

    print "vocab size:",vocabSize
    embeddingTable = np.zeros((vocabSize,EMBEDDING_SIZE),dtype=np.float32)
    # print "embeddingTable shape:",embeddingTable.shape

    for key in w2vDict:
        index = w2vDict[key][0]
        embeddingTable[index] = w2vDict[key][1]

    print "Shape of embedding table:",embeddingTable.shape
    embeddingDict['table'] = embeddingTable
    embeddingDict.close()

def wordsToIndex(words):
    return map( lambda x: w2vDict[x][0],words )

if __name__=='__main__':
    data = pickle.load(open("./tmp/data.txt","r"))
    # allWords = pickle.load(open("./dic.txt","r"))

    # vectorMapping(allWords)
    # logging.info("Mapped all words to vectors!")
    logging.info("Creating newData...")
    newData = map( lambda x: [wordsToIndex(x[0]),x[1]], data)
    logging.info("Saving newData...")
    pickle.dump(newData,open("./tmp/newData.txt","w"))
    # logging.info("Saving EmbeddingTable...")
    # saveEmbeddingTable()
    # logging.info("Done!")

    w2vDict.close()
    tokDict.close()
    