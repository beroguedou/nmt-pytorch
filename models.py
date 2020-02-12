import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *




class Encoder(nn.Module):
    """ """
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super().__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.enc_units, batch_first=True)
        
    def forward(self, input, hidden):
        x = self.embedding(input)
        output, state = self.gru(x, hidden)
        return output, state
        
    def initialize_hidden_state(self):
        return torch.zeros(1, self.batch_sz, self.enc_units)
    
    
    
class Decoder(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, hidden_size):
        super().__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim + hidden_size, self.dec_units, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # used for attention
        self.attention = BahdanauAttention(self.dec_units, hidden_size=hidden_size)
        
    
    def forward(self, input, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing input through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(input)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        context_vector = torch.unsqueeze(context_vector, 1)
        x = torch.cat((context_vector, x), 2)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        
        # output shape == (batch_size * 1, hidden_size)
        output = output.reshape(-1, output.shape[2])

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights
        
