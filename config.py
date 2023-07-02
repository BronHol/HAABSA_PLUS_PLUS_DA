#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys


FLAGS = tf.app.flags.FLAGS
#general variables
tf.app.flags.DEFINE_string('embedding_type', 'BERT','can be: glove, word2vec-cbow, word2vec-SG, fasttext, BERT, BERT_Large, ELMo')
tf.app.flags.DEFINE_integer("year", 2015, "year data set [2015/2016]")
tf.app.flags.DEFINE_string('da_type', 'none','type of data augmentation method (can be: none, EDA-adjusted, BERT, C_BERT, BERT_prepend)') # EDA-adjusted is also implemented, but not considered in this research
tf.app.flags.DEFINE_integer('embedding_dim', 768, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 20, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 100, 'number of train iter')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
tf.app.flags.DEFINE_string('is_r', '1', 'prob')
tf.app.flags.DEFINE_integer('max_target_len', 19, 'max target length')

################################################################################################################################
# HYPERPARAMETERS TUNED IN THIS RESEARCH (FOR REPRODUCING RESEARCH RESULTS, USE THE HYPERPARAMETERS AS SPECIFIED IN README.MD) #
################################################################################################################################
# order of hyperparameters: learning_rate, keep_prob, momentum, l2, batch_size
tf.app.flags.DEFINE_float('learning_rate', 0.08, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.6000000000000001, 'dropout keep prob for the hidden layers of the lcr-rot mode (tuned)')
tf.app.flags.DEFINE_float('keep_prob2', 0.6000000000000001, 'dropout keep prob')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum')
tf.app.flags.DEFINE_float('l2_reg', 0.01, 'l2 regularization')

tf.app.flags.DEFINE_boolean('do_create_raw_files', True, 'whether raw files have to be created, always true when running model for first time') # these three booleans should genreally have the same value (except when troubleshooting)
tf.app.flags.DEFINE_boolean('do_create_augmentation_files', True, 'whether the augmentation file should be made, always true for first time using a DA method')
tf.app.flags.DEFINE_string('augmentation_file_path', 'data/programGeneratedData/'+FLAGS.da_type+'_augmented_data' + str(FLAGS.year)+'.txt', 'augmented train file')

# raw data files
tf.app.flags.DEFINE_string('raw_data_dir', 'data/programGeneratedData/raw_data/', 'folder contataining raw data')
tf.app.flags.DEFINE_string('complete_data_file', FLAGS.raw_data_dir + FLAGS.da_type + '_' +'raw_data'+str(FLAGS.year)+'.txt', 'raw data file for retrieving BERT embeddings, contains both train and test data')
tf.app.flags.DEFINE_string('raw_data_train', FLAGS.raw_data_dir + FLAGS.da_type + '_' + 'raw_data'+str(FLAGS.year)+'_train.txt', 'file raw train data is written to')
tf.app.flags.DEFINE_string('raw_data_test', FLAGS.raw_data_dir + FLAGS.da_type + '_' + 'raw_data'+str(FLAGS.year)+'_test.txt', 'file raw test data is written to')
tf.app.flags.DEFINE_string('raw_data_augmented', FLAGS.raw_data_dir + FLAGS.da_type + '_' + 'raw_data'+str(FLAGS.year)+'_augm.txt', 'file raw augmented data is written to')

# traindata, testdata and embeddings, train path aangepast met ELMo
tf.app.flags.DEFINE_string("train_path_ont", "data/programGeneratedData/GloVetraindata"+str(FLAGS.year)+".txt", "train data path for ont")
tf.app.flags.DEFINE_string("test_path_ont", "data/programGeneratedData/GloVetestdata"+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("train_path", "data/programGeneratedData/" + str(FLAGS.embedding_type) +str(FLAGS.embedding_dim)+'traindata'+str(FLAGS.year)+'_'+str(FLAGS.da_type)+".txt", "train data path")
tf.app.flags.DEFINE_string("test_path", "data/programGeneratedData/" + str(FLAGS.embedding_type) + str(FLAGS.embedding_dim)+'testdata'+str(FLAGS.year)+'_'+str(FLAGS.da_type)+".txt", "formatted test data path")
#tf.app.flags.DEFINE_string("train_path", 'data/programGeneratedData/' + str(FLAGS.embedding_dim) + 'traindata' + str(FLAGS.year) + 'BERT.txt', "train data path")
#tf.app.flags.DEFINE_string("test_path", 'data/programGeneratedData/' + str(FLAGS.embedding_dim) + 'testdata' + str(FLAGS.year) + 'BERT.txt', "formatted test data path")

tf.app.flags.DEFINE_string("embedding_path", "data/programGeneratedData/" + str(FLAGS.embedding_type) + str(FLAGS.embedding_dim)+'embedding'+str(FLAGS.year)+ '_'+ str(FLAGS.da_type)+".txt", "pre-trained glove vectors file path")
#tf.app.flags.DEFINE_string("embedding_path", "data/programGeneratedData/BERT_base.txt", "pre-trained glove vectors file path")

tf.app.flags.DEFINE_string("remaining_test_path_ELMo", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+"ELMo.txt", "only for printing")
tf.app.flags.DEFINE_string("remaining_test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

#toegevoegd vanaf arthur
tf.app.flags.DEFINE_string('bert_embedding_path', 'data/programGeneratedData/bert_embeddings/BERT_base_'+str(FLAGS.da_type) + '_' + str(FLAGS.year)+'.txt', 'path to BERT embeddings file')
tf.app.flags.DEFINE_string('temp_dir', 'data/programGeneratedData/temp/', 'directory for temporary files')
tf.app.flags.DEFINE_string('temp_bert_dir', FLAGS.temp_dir+'bert/', 'directory for temporary BERT files')

# locations for saving BERT finetuning data/external_data
tf.app.flags.DEFINE_string('finetune_train_file', 'data/programGeneratedData/finetuning_data/' + FLAGS.da_type + '_' + str(FLAGS.year)+'_finetune_train.txt', 'file finetuning train data is written to')
tf.app.flags.DEFINE_string('finetune_eval_file', 'data/programGeneratedData/finetuning_data/' + FLAGS.da_type + '_' + str(FLAGS.year)+'_finetune_eval.txt', 'file finetuning evaluation data is written to')
tf.app.flags.DEFINE_string('finetune_model_dir', 'data/programGeneratedData/finetuning_data/' + FLAGS.da_type + '_finetune_model/', 'folder containing BERT model after finetuning')

# Data augmentation vars
tf.app.flags.DEFINE_string("EDA_type", "original", "type of eda (original or adjusted)")
tf.app.flags.DEFINE_integer("EDA_deletion", 1, "number of deletion augmentations")
tf.app.flags.DEFINE_integer("EDA_replacement", 1, "number of replacement augmentations")
tf.app.flags.DEFINE_integer("EDA_insertion", 1, "number of insertion augmentations")
tf.app.flags.DEFINE_integer("EDA_swap", 1, "number of swap augmentations") # in adjusted mode, higher number means more swaps within the same category
tf.app.flags.DEFINE_float("EDA_pct", .2, "percentage of words affected by augmentation") # in adjusted mode EDA_swap not affected

#svm traindata, svm testdata
tf.app.flags.DEFINE_string("train_svm_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'trainsvmdata'+str(FLAGS.year)+".txt", "train data path")
tf.app.flags.DEFINE_string("test_svm_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'testsvmdata'+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("remaining_svm_test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingsvmtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

#hyper traindata, hyper testdata
tf.app.flags.DEFINE_string("hyper_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertraindata'+str(FLAGS.year)+".txt", "hyper train data path")
tf.app.flags.DEFINE_string("hyper_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevaldata'+str(FLAGS.year)+".txt", "hyper eval data path")

tf.app.flags.DEFINE_string("hyper_svm_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertrainsvmdata'+str(FLAGS.year)+".txt", "hyper train svm data path")
tf.app.flags.DEFINE_string("hyper_svm_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevalsvmdata'+str(FLAGS.year)+".txt", "hyper eval svm data path")

#external data sources
tf.app.flags.DEFINE_string("pretrain_file", "data/externalData/"+str(FLAGS.embedding_type)+"."+str(FLAGS.embedding_dim)+"d.txt", "pre-trained embedding vectors for non BERT and ELMo")

tf.app.flags.DEFINE_string("train_data", "data/externalData/restaurant_train_"+str(FLAGS.year)+".xml",
                    "train data path")
tf.app.flags.DEFINE_string("test_data", "data/externalData/restaurant_test_"+str(FLAGS.year)+".xml",
                    "test data path")

tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('prob_file', 'prob1.txt', 'prob')
tf.app.flags.DEFINE_string('saver_file', 'prob1.txt', 'prob')


def print_config():
    #FLAGS._parse_flags()
    FLAGS(sys.argv)
    print('\nParameters:')
    for k, v in sorted(tf.app.flags.FLAGS.flag_values_dict().items()):
        print('{}={}'.format(k, v))


def loss_func(y, prob):
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = - tf.reduce_mean(y * tf.log(prob)) + sum(reg_loss)
    return loss


def acc_func(y, prob):
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc_num, acc_prob


def train_func(loss, r, global_step, optimizer=None):
    if optimizer:
        return optimizer(learning_rate=r).minimize(loss, global_step=global_step)
    else:
        return tf.train.AdamOptimizer(learning_rate=r).minimize(loss, global_step=global_step)


def summary_func(loss, acc, test_loss, test_acc, _dir, title, sess):
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc = tf.summary.scalar('acc' + title, acc)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)
    train_summary_op = tf.summary.merge([summary_loss, summary_acc])
    validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(_dir + '/test')
    validate_summary_writer = tf.summary.FileWriter(_dir + '/validate')
    return train_summary_op, test_summary_op, validate_summary_op, \
        train_summary_writer, test_summary_writer, validate_summary_writer


def saver_func(_dir):
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)
    import os
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return saver
