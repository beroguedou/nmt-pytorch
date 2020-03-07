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
                      
            
            
class LuongAttentionDot(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, query, values): # h_t = values      h_s = query
        query = torch.squeeze(query, 0)
        query = torch.unsqueeze(query, 1)
        query_transposed = query.transpose(2, 1)  
        score = torch.matmul(values, query_transposed) 
        attention_weights = F.softmax(score, dim=1)
        
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights


class LuongAttentionGeneral(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)

        
    def forward(self, query, values):
        query = torch.squeeze(query, 0)
        query = torch.unsqueeze(query, 1)
        query_transposed = query.transpose(2, 1) 

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units) 
        score = torch.matmul(self.W(values), query_transposed)     
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = F.softmax(score, dim=1)
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights


class LuongAttentionConcat(nn.Module):
    
    def __init__(self, units, hidden_size):
        super().__init__()
        self.W = nn.Linear(2 * hidden_size, units)
        self.V = nn.Linear(units, 1)
        
    def forward(self, query, values):

        query = torch.squeeze(query, 0)
        query = torch.unsqueeze(query, 1)
        query = query.repeat(1, 42, 1)
        cat = torch.cat((values, query), dim=2)
        score = self.V(torch.tanh(self.W(cat)))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = F.softmax(score, dim=1)
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights

        
        
        