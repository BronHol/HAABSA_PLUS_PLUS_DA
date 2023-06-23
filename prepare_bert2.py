'''
This Program makes the BERT embedding matrix and test-/traindata, using the tokenisation for BERT
First 'getBERTusingColab' should be used to compile the subfiles containing embeddings.
The new test-/traindata files contain original data, with every word unique and corresponding to vector in emb_matrix
'''
from config import *
from transformers import BertTokenizer

def tokenize_sentence(sentence, tokenizer, word_counts):
    words = "[CLS] " + sentence + " [SEP]"
    tokenized_words = tokenizer.tokenize(words)
    tokens = []
    targetbool = True
    for word in tokenized_words[1:-1]:
      if word == '$':
        if targetbool:
          tokens.append("$T$")
          targetbool = False
      elif word == 't':
        pass
      else:
        count = word_counts.get(word, -1) + 1
        word_counts[word] = count
        token = f'{word}_{count}'
        tokens.append(token)
    return ' '.join(tokens)


#def main():
    '''
    Adds BERT embedding values to sentences in the original test and train datasets. Then
    saves these as separate test and train files, which can be used as an input for a classification
    algorithm.
    '''

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
word_counts = {}

with open('data/programGeneratedData/temp/unique2016_BERT_Data_All.txt', 'w') as output_f:
    lines = open('data/programGeneratedData/raw_data/' + str(FLAGS.da_type) + '_raw_data' + str(FLAGS.year) + '.txt', errors='replace').readlines()

    for i in range(0, len(lines) - 1, 3):
        sentence = lines[i].strip()
        target = lines[i + 1].strip()
        sentiment = lines[i + 2].strip()

        # Tokenize sentence and target
        tokenized_sentence = tokenize_sentence(sentence, tokenizer, word_counts)
        tokenized_target = tokenize_sentence(target, tokenizer, word_counts)

        # Write to output file
        output_f.write(f'{tokenized_sentence}\n')
        output_f.write(f'{tokenized_target}\n')
        output_f.write(f'{sentiment}\n')
    print('Text processing complete. Saved to temporary file')

linesAllData = open('data/programGeneratedData/temp/unique2016_BERT_Data_All.txt').readlines()
with open('data/programGeneratedData/' + str(FLAGS.embedding_dim) + 'traindata' + str(FLAGS.year) + 'BERT.txt',
          'w') as outTrain, \
        open('data/programGeneratedData/' + str(FLAGS.embedding_dim) + 'testdata' + str(FLAGS.year) + 'BERT.txt',
             'w') as outTest:
    for j in range(0, 5640):
        outTrain.write(linesAllData[j])
    for k in range(5640, len(linesAllData)):
        outTest.write(linesAllData[k])
print('Wrote embedding data to train and test files')