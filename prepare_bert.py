# This model prepares BERT embeding files and ccomplete test and train files
# for the HAABSA++ model.
#
# https://github.com/aojudo/HAABSA-plus-plus-DA
#
# Adapted from Van Berkum et al. (2021) https://github.com/stefanvanberkum/CD-ABSC.


# import parameter configuration and data paths
from config import *
import os


def main():
    '''
    Adds BERT embedding values to sentences in the original test and train datasets. Then 
    saves these as separate test and train files, which can be used as an input for a classification
    algorithm. 
    '''
    
    # write all embeddings except for non-words to temporary file
    with open(FLAGS.temp_bert_dir + 'BERT_base_' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + 'embedding.txt', 'w') as out_file:
        with open(FLAGS.bert_embedding_path) as in_file:
            for line in in_file:
                if not (line.startswith('\n') or line.startswith('[CLS]') or line.startswith('[SEP]')):
                    out_file.write(line)

    # write all embeddings except for empty words/newlines to temporary file
    with open(FLAGS.temp_bert_dir + 'BERT_base_' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + 'embedding_withCLS_SEP.txt', 'w') as out_file:
        with open(FLAGS.bert_embedding_path) as in_file:
            for line in in_file:
                if not line.startswith('\n'):
                    out_file.write(line)

    # create table with all unique words from raw dataset
    voca_bert = []
    voca_bert_sep = []
    unique_words = []
    unique_words_index = []
    with open(FLAGS.temp_bert_dir + 'BERT_base_' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + 'embedding_withCLS_SEP.txt') as bert_emb_sep:
        for line in bert_emb_sep:
            word = line.split(' ')[0]
            if not word == '[CLS]':
                voca_bert_sep.append(word)
                if not word == '[SEP]':
                    if word not in unique_words:
                        unique_words.append(word)
                        unique_words_index.append(0)
                    voca_bert.append(word)
    print('vocaBERT: ' + str(len(voca_bert)))
    print('voca_bert_sep: ' + str(len(voca_bert_sep)))

    # create embedding matrix with unique words, print counter
    counter = 0
    unique_voca_bert = []
    with open(FLAGS.temp_bert_dir + 'BERT_base_' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + 'embedding.txt') as bert_emb:
        with open(FLAGS.embedding_path, 'w') as out_file:
            for line in bert_emb:
                word = line.split(' ')[0]
                counter += 1
                weights = line.split(' ')[1:]
                index = unique_words.index(word)  # get index in unique words table.
                word_count = unique_words_index[index]
                unique_words_index[index] += 1
                item = str(word) + '_' + str(word_count)
                out_file.write('%s ' % item)
                unique_voca_bert.append(item)
                first = True
                for weight in weights[:-1]:
                    out_file.write('%s ' % weight)
                out_file.write('%s' % weights[-1])

    # make uniqueBERT_SEP variable
    unique_voca_bert_sep = []
    counti = 0
    sepcounterunique = 0
    for i in range(0, len(voca_bert_sep)):
        if voca_bert_sep[i] == '[SEP]':
            sepcounterunique += 1
            unique_voca_bert_sep.append('[SEP]')
        else:
            unique_voca_bert_sep.append(unique_voca_bert[counti])
            counti += 1
    print('voca_bert_sep: ' + str(len(voca_bert_sep)))
    print('uniqueVocaBERT: ' + str(len(unique_voca_bert)))
    print('unique_voca_bert_sep: ' + str(len(unique_voca_bert_sep)))
    print('sepcounterunique: ' + str(sepcounterunique))

    # make a matrix (three vectors) containing for each word in bert-tokeniser 
    # style the word_id (x_word), sentence_id (x_sent), target boolean, (x_targ)
    lines = open(FLAGS.raw_data_file).readlines()
    print('number of lines in lines is: ' + str(len(lines)))
    index = 0
    index_sep = 0
    x_word = []
    x_sent = []
    x_targ = []
    x_tlen = []
    sentence_count = 0
    target_raw = []
    sentiment = []
    targets_insent = 0
    for i in range(0, len(lines), 3):
        target_raw.append(lines[i + 1].lower().split())
        sentiment.append(lines[i + 2])
    for i in range(0, len(voca_bert_sep)):
        sentence_target = target_raw[sentence_count]
        sentence_target_str = ''.join(sentence_target)
        x_word.append(i)
        word = voca_bert_sep[i]
        x_sent.append(sentence_count)
        x_tlen.append(len(sentence_target))
        if word == '[SEP]':
            sentence_count += 1
            i_new_sent = i + 1
        tar_guess = ''
        for j in range(len(sentence_target) - 1, -1, -1):
            if voca_bert_sep[i - j][:2] == '##':
                tar_guess += voca_bert_sep[i - j][2:]
            else:
                tar_guess += voca_bert_sep[i - j]
        if tar_guess == sentence_target_str:
            x_targ.append(1)
            for k in range(0, len(sentence_target)):
                x_targ[i - k] = 1
        else:
            x_targ.append(0)

    # print to BERT data to text file
    for filenr in range(1, 8):
        sentence_senten_unique = ''
        sentence_target_unique = ''
        sent_count = 0
        dollar_count = 0
        with open(FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_' + str(filenr) + '.txt', 'w') as out_file:
            for u in range(0, len(unique_voca_bert_sep)):
                if unique_voca_bert_sep[u] == '[SEP]':
                    out_file.write(sentence_senten_unique + '\n')
                    out_file.write(sentence_target_unique + '\n')
                    out_file.write(''.join(sentiment[sent_count]))
                    sentence_senten_unique = ''
                    sentence_target_unique = ''
                    sent_count += 1
                else:
                    if x_targ[u] == 1:
                        dollar_count += 1
                        if dollar_count == 1:
                            sentence_senten_unique += '$T$ '
                        sentence_target_unique += unique_voca_bert_sep[u] + ' '
                    else:
                        dollar_count = 0
                        sentence_senten_unique += unique_voca_bert_sep[u] + ' '

        lines = open(FLAGS.raw_data_file).readlines()
        index = 0
        index_sep = 0
        x_word = []
        x_sent = []
        x_targ = []
        x_tlen = []
        sentence_count = 0
        target_raw = []
        sentiment = []
        targets_insent = 0
        for i in range(0, len(lines), 3):
            target_raw.append(lines[i + 1].lower().split())
            sentiment.append(lines[i + 2])
        for i in range(0, len(voca_bert_sep)):
            sentence_target = target_raw[sentence_count]
            sentence_target_str = ''.join(sentence_target)
            x_word.append(i)
            word = voca_bert_sep[i]
            x_sent.append(sentence_count)
            x_tlen.append(len(sentence_target))
            if word == '[SEP]':
                sentence_count += 1
                i_new_sent = i + 1
            tar_guess = ''
            for j in range(len(sentence_target) - 1 + filenr, -1, -1):
                if voca_bert_sep[i - j][:2] == '##':
                    tar_guess += voca_bert_sep[i - j][2:]
                else:
                    tar_guess += voca_bert_sep[i - j]
            if tar_guess == sentence_target_str:
                x_targ.append(1)
                for k in range(0, len(sentence_target) + filenr):
                    x_targ[i - k] = 1
            else:
                x_targ.append(0)

    # Combine words, this is needed for different tokenisation for target phrase.
    # Different files for different extra target lengths, e.g. file 2 contains 
    # target phrases that are 1 word longer in the BERT embedding than the original
    # target phrase (Comment by M. Trusca, 
    # https://github.com/mtrusca/HAABSA_PLUS_PLUS).
    lines_1 = open(FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_1.txt').readlines()
    lines_2 = open(FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_2.txt').readlines()
    lines_3 = open(FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_3.txt').readlines()
    lines_4 = open(FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_4.txt').readlines()
    lines_5 = open(FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_5.txt').readlines()
    lines_6 = open(FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_6.txt').readlines()
    lines_7 = open(FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_7.txt').readlines()

    # troubleshooting print
    print('number of lines in lines_1 is: '+str(len(lines_1)))

    line_count = 0
    with open(FLAGS.temp_bert_dir + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_All.txt', 'w') as out_f:
        for i in range(0, len(lines_1), 3):
            if lines_1[i + 1] == '\n':
                if lines_2[i + 1] == '\n':
                    if lines_3[i + 1] == '\n':
                        if lines_4[i + 1] == '\n':
                            if lines_5[i + 1] == '\n':
                                if lines_6[i + 1] == '\n':
                                    out_f.write(lines_7[i])
                                    out_f.write(''.join(lines_7[i + 1]))
                                else:
                                    out_f.write(lines_6[i])
                                    out_f.write(''.join(lines_6[i + 1]))
                            else:
                                out_f.write(lines_5[i])
                                out_f.write(''.join(lines_5[i + 1]))
                        else:
                            out_f.write(lines_4[i])
                            out_f.write(''.join(lines_4[i + 1]))
                    else:
                        out_f.write(lines_3[i])
                        out_f.write(''.join(lines_3[i + 1]))
                else:
                    out_f.write(lines_2[i])
                    out_f.write(''.join(lines_2[i + 1]))
            else:
                out_f.write(lines_1[i])
                out_f.write(''.join(lines_1[i + 1]))
            
            out_f.write(lines_1[i + 2])
            line_count += 1
        
        ## retrive number of lines in training data
        # with open(FLAGS.raw_data_train, 'r') as file:
            # train_lines = len(file.read().splitlines())
        
        ## split in train and test file
        # lines_all_data = out_f.readlines()
        
        # troubleshooting print
        print('number of items in lines_all_data is: '+str(line_count))
    
    with open(FLAGS.temp_bert_dir + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_All.txt', 'r') as file:
        lines_all_data = file.readlines()
        print('number of lines in lines_all_data is: '+str(len(lines_all_data)))
        
        #train_lines = sum(1 for line in open(FLAGS.raw_data_train))
        
        with open(FLAGS.train_path,'w') as out_train:
            #range veranderen als augementaties maken
            for j in range(0, 5630):
                out_train.write(lines_all_data[j])
        print ('Succesfully created BERT train file at '+str(FLAGS.train_path))
    
        with open(FLAGS.test_path,'w') as out_test:
            # range veranderen als augementaties maken
            for k in range(5630, len(lines_all_data)):
                out_test.write(lines_all_data[k])
        print ('Succesfully created BERT test file at '+str(FLAGS.test_path))
        
    # remove all temporary files that have been created
    #for file in [FLAGS.temp_bert_dir + 'BERT_base_' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + 'embedding.txt',
     #            FLAGS.temp_bert_dir + 'BERT_base_' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + 'embedding_withCLS_SEP.txt',
      #           FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_' + str(filenr) + '.txt',
       #          FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_1.txt',
        #         FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_2.txt',
         #        FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_3.txt',
          #       FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_4.txt',
           #      FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_5.txt',
            #     FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_6.txt',
             #    FLAGS.temp_bert_dir + 'unique' + '_' + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_7.txt',
              #   FLAGS.temp_bert_dir + FLAGS.da_type + '_' + str(FLAGS.year) + '_BERT_Data_All.txt']:
        #if os.path.exists(file):
         #   os.remove(file)


if __name__ == '__main__':
    main()
