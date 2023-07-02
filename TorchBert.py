''' Run the following code in Google Colab to obtain the BERT embeddings. Therefore, you need to upload the raw data files containing the train and test dataset.
    Perform this step after you made the data augmentations.'''

import pandas as pd
import numpy as np
import torch

from google.colab import files

pip install transformers

from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained('bert-base-uncased',
           output_hidden_states = True,)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_text_preparation(text, tokenizer):
  """
  Preprocesses text input in a way that BERT can interpret.
  """
  marked_text = "[CLS] " + text + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1]*len(indexed_tokens)
# convert inputs to tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensor = torch.tensor([segments_ids])
  return tokenized_text, tokens_tensor, segments_tensor

def get_bert_embeddings(tokens_tensor, segments_tensor, model):
    """
    Obtains BERT embeddings for tokens.
    """
    # gradient calculation id disabled
    with torch.no_grad():
      # obtain hidden states
      outputs = model(tokens_tensor, segments_tensor)
      hidden_states = outputs[2]
    # concatenate the tensors for all layers
    # use "stack" to create new dimension in tensor
    token_embeddings = torch.stack(hidden_states, dim=0)
    # remove dimension 1, the "batches"
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # swap dimensions 0 and 1 so we can loop over tokens
    token_embeddings = token_embeddings.permute(1,0,2)
    # intialized list to store embeddings
    token_vecs_sum = []
    # "token_embeddings" is a [Y x 12 x 768] tensor
    # where Y is the number of tokens in the sentence
    # loop over tokens in sentence
    for token in token_embeddings:
    # "token" is a [12 x 768] tensor
    # sum the vectors from the last four layers
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum

upload = files.upload()  # raw2016forBERT

#get the number of lines in the file
lines = open('none_raw_data2015.txt', errors='replace').readlines()
print(len(lines)/3)

from collections import OrderedDict
context_embeddings = []
context_tokens = []
# Change outfile name to embedding_path in config
lines = open('none_raw_data2015.txt', errors='replace').readlines()
with open('BERT768embedding2015_none.txt', 'w', encoding='utf-8') as f:
    word_counts = {}
    for i in range(0 * 3, 1880 * 3, 3):  # len(lines): 2530 for 2016, 4410 for BERT-models, 8170 for EDA-adjusted, 10050 for EDA-original
        print("sentence: " + str(i / 3) + " out of " + str(len(lines) / 3) + " in " + "raw_data;")
        target = lines[i + 1].lower().split()
        words = lines[i].lower().split()
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                words_l.append(word)
            else:
                words_r.append(word)
        sentence = " ".join(words_l + target + words_r)
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
        # make ordered dictionary to keep track of the position of each   word
        tokens = OrderedDict()


        # loop over tokens in sensitive sentence
        for token in tokenized_text[1:-1]:
          # keep track of position of word and whether it occurs multiple times
          if token in tokens:
            tokens[token] += 1
          else:
            tokens[token] = 1
          count = word_counts.get(token, -1) + 1
          word_counts[token] = count
        # compute the position of the current token
          token_indices = [i for i, t in enumerate(tokenized_text) if t == token]
          current_index = token_indices[tokens[token]-1]
        # get the corresponding embedding
          token_vec = list_token_embeddings[current_index]
          token_vec_array = token_vec.numpy()

        # save values
        #  context_tokens.append(token)
        #  context_embeddings.append(token_vec)

          f.write('\n%s_%s ' % (token, count))
          f.write(' '.join(map(str, token_vec_array)))

#Change filename to file for download
files.download('BERT768embedding2015_none.txt')