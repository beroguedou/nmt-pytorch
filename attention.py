import torch
import torch.nn as nn
import torch.nn.functional as F




class BahdanauAttention(nn.Module):
    
    def __init__(self, units, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, units)
        self.W2 = nn.Linear(hidden_size, units)
        self.V = nn.Linear(units, 1)
        
    def forward(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        query = torch.squeeze(query, 0)
        hidden_with_time_axis = torch.unsqueeze(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        sum_1 = self.W1(values) + self.W2(hidden_with_time_axis)
        score = self.V(torch.tanh(sum_1))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = F.softmax(score, dim=1)
        
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights
                      
            
        
        
        