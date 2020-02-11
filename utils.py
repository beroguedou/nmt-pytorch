
import os 
import io
import re
import sys
import unicodedata
from tqdm import tqdm
import requests
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


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)