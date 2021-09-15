import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import torch.nn.functional as F
import math


class Wavenet(nn.Module):
    def __init__(self,num_classes,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels = 256,
                 end_channels = 256,
                 kernel_size= 2,blocks = 4,
                 layers = 4,
                 embedding_dimension = 100):
        super(Wavenet, self).__init__()
        self.layers = layers
        self.blocks = blocks
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.relu = torch.nn.ReLU(inplace=False)
        self.Sigmoid = torch.nn.Sigmoid()
        self.Tanh = torch.nn.Tanh()
        self.convblocks = [];
        self.embedding_dimension = embedding_dimension
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dimension)

        self.init_conv1d = torch.nn.Conv1d(in_channels=self.embedding_dimension,
                                           out_channels=self.residual_channels,
                                           kernel_size=1)
        
        self.end_conv1d = nn.Conv1d(in_channels=self.skip_channels,
                                  out_channels=self.end_channels,
                                  kernel_size=1,
                                  bias=True)
        
        '''Added this extra layer'''
        self.end_conv2d = torch.nn.Conv1d(in_channels=self.end_channels,
                                           out_channels=self.end_channels,
                                           kernel_size=1)
        
     
        '''Fully connected layers'''
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, self.num_classes)


        for i in range(blocks):
          new_dilation = 1
          current_block = []
          for j in range(layers):

            dilated_conv = torch.nn.Conv1d(in_channels=self.residual_channels, out_channels=self.dilation_channels,kernel_size=  self.kernel_size,dilation = new_dilation , padding = 0)

            filter_conv =  torch.nn.Conv1d(in_channels=self.dilation_channels, out_channels=self.dilation_channels, kernel_size=self.kernel_size)

            gate_conv =  torch.nn.Conv1d(in_channels=self.dilation_channels, out_channels=self.dilation_channels, kernel_size=self.kernel_size)

            residual_conv = torch.nn.Conv1d(in_channels=self.dilation_channels, out_channels=self.residual_channels,kernel_size = 1) 

            skip_conv = torch.nn.Conv1d(in_channels=self.dilation_channels, out_channels=self.skip_channels, kernel_size = 1)

            current_layer = {
                'dilated_conv':dilated_conv,
                'filter_conv':filter_conv,
                'gate_conv':gate_conv,
                'residual_conv':residual_conv,
                'skip_conv':skip_conv,
                'new_dilation':new_dilation,
            }
            current_block.append(current_layer)
            new_dilation *= 2
          self.convblocks.append(current_block)
            
       

    def forward(self, input):
      x = self.embedding(input)
      x = x.permute(0,2,1)
      x = self.init_conv1d(x)
      skip = 0
      for i in range(self.blocks):
        for j in range(self.layers):
          # print('dilation:  ',self.convblocks[i][j]['new_dilation'] )
          x = F.pad(x,(self.convblocks[i][j]['new_dilation']+1,0), mode='constant', value=0)
          # print('INPUT :block '+str(i)+'   layer : '+str(j) + '   shape :',x.shape)
          residual = self.convblocks[i][j]['dilated_conv'](x)
          # print('block '+str(i)+'   layer : '+str(j) + '   shape :',residual.shape)
          filter = self.convblocks[i][j]['filter_conv'](residual)
          filter = self.Tanh(filter)
          gate = self.convblocks[i][j]['gate_conv'](residual)
          gate = self.Sigmoid(gate)

          x = filter*gate
           
          skip = self.convblocks[i][j]['skip_conv'](x)
          x = self.convblocks[i][j]['residual_conv'](x) 
          x = x+ residual[:,:,:-(self.kernel_size-1)]
          # print('block '+str(i)+'   layer : '+str(j) + '   shape :',x.shape)
      
      wavenet_output = self.end_conv2d(self.relu(self.end_conv1d(self.relu(skip))))
      output = self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(wavenet_output[:,:,-1])))))))
      # output = self.fc2(self.relu(self.fc1(wavenet_output)))
   
      return output


    
