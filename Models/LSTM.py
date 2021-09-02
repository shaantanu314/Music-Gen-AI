import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Sequence(nn.Module): 
    def __init__(self,num_classes, seq_length,embedding_dimension = 100,num_layers= 3,hidden_size_lstm = 256):
        super(Sequence, self).__init__()
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.hidden_size_lstm = hidden_size_lstm
        self.embedding_dimension = embedding_dimension
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dimension).to(device)
        self.lstm = nn.LSTM(input_size= self.embedding_dimension, hidden_size=self.hidden_size_lstm ,num_layers=self.num_layers,batch_first = True).to(device)
        self.linear1 = nn.Linear(self.hidden_size_lstm,128).to(device)
        self.relu = torch.nn.ReLU(inplace=False).to(device)
        self.linear2 = nn.Linear(128,128) .to(device)
        self.softmax = torch.nn.Softmax().to(device)
        self.linear3 = nn.Linear(128,self.num_classes).to(device)
        #self.linear4 = nn.Linear(64,self.num_classes) 
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, input ,batch_size):
      x = self.embedding(input)
    #   print(device)
      h_t = torch.zeros(self.num_layers,batch_size, 256, dtype=torch.float).to(device)
      c_t = torch.zeros(self.num_layers,batch_size, 256, dtype=torch.float).to(device)
      x,_ = self.lstm(x,(h_t,c_t))
      # print(x.shape)
      lstm_output = x[:,-1,:]
      # print(lstm_output.shape)
      output = self.linear3(self.relu(self.dropout(self.linear2(self.relu(self.dropout(self.linear1(self.relu(self.dropout(lstm_output)))))))))
      # output = self.softmax(output)
      return output

   