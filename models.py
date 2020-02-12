import torch
import torch.nn as nn
import torch.nn.functional as F
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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
    
    
    
                          