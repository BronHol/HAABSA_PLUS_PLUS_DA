# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC

import tensorflow as tf
# import cabascModel
#import lcrModel
#import lcrModelInverse
#import lcrModelAlt
#import svmModel
from OntologyReasoner import OntReasoner
from loadData import *

#import parameter configuration and data paths
from config import *

#import modules
import numpy as np
import sys

import lcrModelAlt_hierarchical_v1
import lcrModelAlt_hierarchical_v2
import lcrModelAlt_hierarchical_v3
import lcrModelAlt_hierarchical_v4

# main function
def main(_):
    loadData         = False        # Only True for making data augmentations or raw_data files
                                    # Use TorchBert in Google Colab to generate the BERT embeddings for every word
                                    # Use prepare_bert for making train and test data sets
    useOntology      = False        # When run together with runLCRROTALT, the two-step method is used
    shortCutOnt      = True         # Only possible when last run was for same year
    runLCRROTALT     = False

    runSVM           = False
    runCABASC        = False
    runLCRROT        = False
    runLCRROTINVERSE = False
    weightanalysis   = False

    runLCRROTALT_v1     = False
    runLCRROTALT_v2     = False
    runLCRROTALT_v3     = False
    runLCRROTALT_v4     = True

    #determine if backupmethod is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM or runLCRROTALT_v1 or runLCRROTALT_v2 or runLCRROTALT_v3 or runLCRROTALT_v4:
        backup = True
    else:
        backup = False

    da_methods = FLAGS.da_type.split('-')
    da_type = da_methods[0]
    adjusted = False
    if da_type == 'EDA':
        use_eda = True
        if len(da_methods) > 1:
            if da_methods[1] == 'adjusted':
                adjusted = True
            else:
                raise Exception('The EDA type used in FLAGS.da_type.split does not exist. Please correct flag value.')
        else:
            raise Exception('The EDA type to use is not specified. Please complete flag value.')
    else:
        use_eda = False

    # determine whether bert should be used for DA
    use_bert = False
    if FLAGS.da_type == 'BERT':
        use_bert = True

    # determine whether bert-prepend should be used for DA
    use_bert_prepend = False
    if FLAGS.da_type == 'BERT_prepend':
        use_bert_prepend = True

    # determine whether c-bert should be used for DA
    use_c_bert = False
    if FLAGS.da_type == 'C_BERT':
        use_c_bert = True

    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData, use_eda, adjusted, use_bert, use_bert_prepend, use_c_bert)
    print(test_size)
    remaining_size = 250
    accuracyOnt = 0.87

    if useOntology == True:
        print('Starting Ontology Reasoner')
        #in sample accuracy
        Ontology = OntReasoner()
        accuracyOnt, remaining_size = Ontology.run(backup,FLAGS.test_path_ont, runSVM)
        #out of sample accuracy
        #Ontology = OntReasoner()      
        #accuracyInSampleOnt, remainingInSample_size = Ontology.run(backup,FLAGS.train_path_ont, runSVM)        
        if runSVM == True:
            test = FLAGS.remaining_svm_test_path
        else:
            test = FLAGS.remaining_test_path
            print(test[0])
        print('train acc = {:.4f}, test acc={:.4f}, remaining size={}'.format(accuracyOnt, accuracyOnt, remaining_size))
    else:
        if shortCutOnt == True:
            #2015
            #accuracyOnt = 0.8277
            #remaining_size = 301
            #2016
            accuracyOnt = 0.8682
            remaining_size = 248
            test = FLAGS.remaining_test_path
        else:
            test = FLAGS.test_path

    # LCR-Rot-hop model
   # if runLCRROTALT == True:
    #   _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt.main(FLAGS.train_path, test, accuracyOnt, test_size,
      #                                                  remaining_size)
     #  tf.reset_default_graph()

    if runLCRROTALT_v1 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v1.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size)
       tf.reset_default_graph()

    if runLCRROTALT_v2 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v2.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size)
       tf.reset_default_graph()

    if runLCRROTALT_v3 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v3.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size)
       tf.reset_default_graph()

    if runLCRROTALT_v4 == True:
       _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                        remaining_size)
       tf.reset_default_graph()

'''
    # LCR-Rot model
    #if runLCRROT == True:
    #    _, pred1, fw1, bw1, tl1, tr1, sent, target, true = lcrModel.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
    #    tf.reset_default_graph()

    # LCR-Rot-inv model
    #if runLCRROTINVERSE == True:
    #    lcrModelInverse.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
    #   tf.reset_default_graph()    

    # CABASC model
    if runCABASC == True:
        _, pred3, weights = cabascModel.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
        if weightanalysis and runLCRROT and runLCRROTALT:
            outF= open('sentence_analysis.txt', "w")
            dif = np.subtract(pred3, pred1)
            for i, value in enumerate(pred3):
                if value == 1 and pred2[i] == 0:
                    sentleft, sentright = [], []
                    flag = True
                    for word in sent[i]:
                        if word == '$t$':
                            flag = False
                            continue
                        if flag:
                            sentleft.append(word)
                        else:
                            sentright.append(word)
                    print(i)
                    outF.write(str(i))
                    outF.write("\n")
                    outF.write('lcr pred: {}; CABASC pred: {}; lcralt pred: {}; true: {}'.format(pred1[i], pred3[i], pred2[i], true[i]))
                    outF.write("\n")
                    outF.write(";".join(sentleft))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in fw1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sentright))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in bw1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(target[i]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tl1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tr1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sentleft))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in fw2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sentright))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in bw2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(target[i]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tl2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tr2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sent[i]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in weights[i][0]))
                    outF.write("\n")
            outF.close()

    # BoW model
    if runSVM == True:
        svmModel.main(FLAGS.train_svm_path,test, accuracyOnt, test_size, remaining_size)
'''
print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
