import os 
import io
import re
import sys
import torch
import random
import requests
import unicodedata
import tensorflow as tf
from tqdm import tqdm
from zipfile import ZipFile




url_base = 'http://storage.googleapis.com/download.tensorflow.org/data/'
file_name = 'spa-eng.zip'


def get_file(url_base=url_base, file_name=file_name, extract=True):
    
    url = url_base + file_name
    response = requests.get(url)
    #path = file_name[:file_name.find('.zip')]
    with open(file_name, 'wb') as fic:
        fic.write(response.content)
    with ZipFile(file_name, 'r') as zipObj:
       # Extract all the contents of zip file in the current directory
       zipObj.extractall()
    # delete the zip file
    os.system('rm {}'.format(file_name))
    
    print('======== THE DATASET IS READY ========')
 

    
# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
          if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w



def create_dataset(path, num_examples):
    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    print("The number of line in the dataset is", len(lines))
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    random.shuffle(word_pairs)
    
    return zip(*word_pairs)


def max_length(tensor):
      return max(len(t) for t in tensor)

    
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def convert(lang, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, lang.index_word[t]))

            
            
def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, device, batch_sz, targ_lang, teacher_forcing_ratio=0.5):
    # Initialize the encoder
    encoder_hidden = encoder.initialize_hidden_state().to(device)
    # Put all the previously computed gradients to zero
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(1)
    
    # Encode the input sentence
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    loss = 0
    decoder_input = torch.tensor([[targ_lang.word_index['<start>']]] * batch_sz, device=device)
    decoder_hidden = encoder_hidden

    # Use randomly teacher forcing
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True  
    else:  
        use_teacher_forcing = False

    #if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input to help the model
    # in case it starts with the wrong word.
    for di in range(1, target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[:, di])
        if use_teacher_forcing:
            decoder_input = torch.unsqueeze(target_tensor[:, di], 1)  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.data.topk(1)
            # the predicted ID is fed back into the model
            decoder_input = topi.detach()

    
    batch_loss = (loss.item() / int(target_tensor.shape[1]))
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return batch_loss



def evaluate(sentence, max_length_targ, max_length_inp, encoder, decoder, inp_lang, targ_lang, device):

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = torch.tensor(inputs).long().to(device)

    result = ''

    with torch.no_grad():
        hidden = torch.zeros(1, 1, 1024, device=device)
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = torch.tensor([[targ_lang.word_index['<start>']]], device=device)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

            # storing the attention weights to plot later on
            topv, topi = predictions.data.topk(1)
            result += targ_lang.index_word[topi.item()] + ' '

            if targ_lang.index_word[topi.item()] == '<end>':
                return result, sentence

            # the predicted ID is fed back into the model

            dec_input = torch.tensor([topi.item()], device=device).unsqueeze(0)


        return result, sentence #, attention_plot


def translate(sentence, max_length_targ, max_length_inp, encoder, decoder, inp_lang, targ_lang, device):
    result, sentence = evaluate(sentence, max_length_targ, max_length_inp, 
                                                encoder, decoder, inp_lang, targ_lang, device)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))