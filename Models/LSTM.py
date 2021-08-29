import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import torch.nn.functional as F
import math


class Sequence(nn.Module): 
    def __init__(self,num_classes, seq_length,embedding_dimension = 100):
        super(Sequence, self).__init__()
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.embedding_dimension = embedding_dimension
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dimension)
        self.lstm = nn.LSTM(input_size= self.embedding_dimension, hidden_size=128 ,num_layers=3,batch_first = True)
        self.linear1 = nn.Linear(128,256)
        self.relu = torch.nn.ReLU(inplace=False)
        self.linear2 = nn.Linear(256,self.num_classes) 
        self.softmax = torch.nn.Softmax()
        # self.linear3 = nn.Linear(64,32)
        #self.linear4 = nn.Linear(64,self.num_classes) 
        # self.dropout = nn.Dropout(p=0.3)


    def forward(self, input ,batch_size):
      x = self.embedding(input)
      h_t = torch.zeros(3,batch_size, 128, dtype=torch.float)
      c_t = torch.zeros(3,batch_size, 128, dtype=torch.float)
      x,_ = self.lstm(x,(h_t,c_t))
      # print(x.shape)
      lstm_output = x[:,-1,:]
      # print(lstm_output.shape)
      output = self.linear2(self.relu(self.linear1(self.relu(lstm_output))))
      # output = self.softmax(output)
      return output

    def predict(self,input,steps=1):
        outputs = []
        curr_window = input
        # curr_window = torch.zeros(1,input.shape[0],dtype=torch.int32);      
        # curr_window[0,:] = torch.IntTensor(input) 
        with torch.no_grad():
          for i in range(steps):
            output = self.forward(curr_window,curr_window.shape[0])
            ind = torch.argmax(output)
            # print(output[ind])
            # print(output)
            ind_ = ind.numpy()
            outputs.append(ind_)
            # print(outputs)
            next = torch.empty(1,1);
            next[0,:] = ind
            # print(curr_window)
            # print(output)
            curr_window = torch.cat((curr_window,next),1)
            curr_window = curr_window[:,1:]
            curr_window = curr_window.type(torch.int32)
            
            # plt.figure()
            # plt.plot(curr_window[0,:,0])

        return outputs
    