'''Run the following code in Google Colab to make the fine-tuned models. Therefore, you need to upload the finetune train and eval files. Also the python file run_lm_finetuning.py should be uploaded to session directory. Then you need to download the model files to your directory for the unmasker.'''

from google.colab import files

upload = files.upload()

raw_train_file = 'GloVetraindata2016.txt'
finetune_train_file = 'BERT_2016_finetune_train.txt'
finetune_eval_file = 'BERT_2016_finetune_eval.txt'

import random as rd

def prepare_data(raw_train_file, finetune_train_file, finetune_eval_file):
    '''
    Takes raw train data and turns it into a train and eval file containing
    label-sentence combinations.

    :param raw_train_file: file containing raw train data
    :param finetune_train_file: file for saving finetune train data
    :param finetune_eval_file: file for saving finetune eval data
    '''

    rd.seed(12345)

    with open(raw_train_file, 'r') as in_f, open(finetune_train_file, 'w+', encoding='utf-8') as out_train, open(finetune_eval_file, 'w+', encoding='utf-8') as out_eval:
        lines = in_f.readlines()
        for i in range(0, len(lines)-1, 3):
            sentence = lines[i]

            # randomly split into train and test data (80/20 split)
            if rd.random() < 0.8:
                out_train.writelines([sentence])
            else:
                out_eval.writelines([sentence])

prepare_data(raw_train_file, finetune_train_file, finetune_eval_file)

files.download(finetune_train_file)

'''!python run_lm_finetuning.py \
               --output_dir='/content/drive/MyDrive/BronScriptie' \
               --model_type=bert \
               --model_name_or_path=bert-base-uncased \
               --do_train \
               --train_data_file='/content/BERT_2016_finetune_train.txt' \
               --do_eval \
               --eval_data_file='/content/BERT_2016_finetune_eval.txt' \
               --mlm \
               --per_gpu_train_batch_size=1 \
               --per_gpu_eval_batch_size=1 \
               --num_train_epochs=20 \
               --save_total_limit=1'''