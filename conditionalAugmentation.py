import string
from transformers import BertTokenizer, BertForMaskedLM
from transformers import pipeline
import torch
import re
import random as rd

BERT_MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
unmasker = pipeline(task='fill-mask', model='data/programGeneratedData/finetuning_data/BERT_finetune_model/', tokenizer='bert-base-uncased',topk = 2)
rd.seed(546297)

model = BertForMaskedLM.from_pretrained(BERT_MODEL)


def file_maker_conditional(in_file, out_file):
    print('Starting C-BERT-augmentation')
    with open(in_file, 'r') as in_f, open(out_file, 'w', encoding='utf-8') as out_f:
        lines = in_f.readlines()
        for i in range(0, len(lines) - 1, 3):
            print(i)
            old_sentence = lines[i].strip()
            target = lines[i + 1].strip()
            sentiment = lines[i + 2].strip()
            new_sentence = augment_sentence_conditional(old_sentence,target,sentiment)
            out_f.writelines([old_sentence + '\n', target + '\n', sentiment + '\n'])
            out_f.writelines([new_sentence + '\n', target + '\n', sentiment + '\n'])
    return out_file

def augment_sentence_conditional(in_sentence, in_target, in_sentiment):

    words = tokenizer.tokenize(in_sentence)
    tar = re.findall(r'\w+|[^\s\w]+', in_target)
    for word in tar:
        word = tokenizer.tokenize(word)
    tar_length = len(tar)

    masked_sen = []
    ind = 0

    if in_sentiment == '-1':
        sentiment = 'negative'
    elif in_sentiment == '0':
        sentiment = 'neutral'
    elif in_sentiment == '1':
        sentiment = 'positive'
    else:
        print('no correct sentiment')


    for wrd in words:
        j = words.index(wrd)
        if wrd == '$' and words[j+1]=='t' and words[j+2]=='$':
            ind = words.index(wrd)

    masked_sen.extend(words[:ind])
    masked_sen.extend(tar)
    masked_sen.extend(words[(ind+3):])
    augmentend_sentence = []

    j=0
    number_not_words = 0
    while j < len(words):
        if words[j] == '$' and words[j+1]=='t' and words[j+2]=='$':
            j+=3
            number_not_words +=3
        elif words[j] in string.punctuation:
            j += 1
            number_not_words +=1
        else:
            j += 1

    mask_prob = 0.15
    total_masks = min(1,int(round((len(words)-number_not_words)*mask_prob)))
    amount_masked = 0
    vocab = tokenizer.vocab
    real_percentage = mask_prob / ( (len(words)-number_not_words)/len(words) )

    i=0
    while i < len(words):
        if words[i] == '$' and words[i+1]=='t' and words[i+2]=='$':
            augmentend_sentence.append('$T$')
            i+=3
        elif words[i] in string.punctuation:
            augmentend_sentence.append(words[i])
            i += 1
        else:
            prob1 = rd.random()
            if prob1 <= real_percentage:
                prob2 = rd.random()
                if prob2 <= 0.8:
                    amount_masked += 1
                    cur_sent = masked_sen.copy()
                    masked_word = words[i]
                    if i < ind:
                        cur_sent[i] = '[MASK]'
                    else:
                        cur_sent[i-(3-tar_length)] = '[MASK]'
                    tokens = ["[CLS]"] + cur_sent + ["SEP"]
                    masked_index = tokens.index('[MASK]')
                    if sentiment =='positive':
                        token_type_ids = torch.tensor([0]*len(tokens)).unsqueeze(0)
                    elif sentiment == 'neutral':
                        token_type_ids = torch.tensor([1]*len(tokens)).unsqueeze(0)
                    else:
                        token_type_ids = torch.tensor([1] * len(tokens)).unsqueeze(0)

                    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)

                    with torch.no_grad():
                        outputs = model(input_ids, token_type_ids=token_type_ids)

                    predictions = outputs[0][0, masked_index].topk(k=2).indices.tolist()
                    predicted_words = tokenizer.convert_ids_to_tokens(predictions)

                    if predicted_words[0] == masked_word:
                        augmentend_sentence.append(predicted_words[1])
                        i += 1
                    else:
                        augmentend_sentence.append(predicted_words[0])
                        i += 1
                elif 0.8 < prob2 <= 0.9:
                    amount_masked += 1
                    random_token = rd.choice(list(vocab.keys()))
                    augmentend_sentence.append(random_token)
                    i += 1
                else:
                    augmentend_sentence.append(words[i])
                    i += 1
            else:
                augmentend_sentence.append(words[i])
                i+=1

    augmentend_sentence_str = ' '.join(augmentend_sentence)
    return augmentend_sentence_str

def is_punctuation(word):
    return all(char in string.punctuation for char in word)