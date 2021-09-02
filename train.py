from music21 import *
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import Generate_dataset
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import torch.nn.functional as F
import math
from MusicDataset import *
from Models import Wavenet,LSTM
import torch.optim as optim
import time , sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

baseDir = '/home/god-particle/Desktop/Music_Gen_AI'
expDir = os.path.join(baseDir, 'trained_model_cache', time.strftime("%d_%m_%Y_%H_%M"))
lossDir = os.path.join(expDir, 'loss')
os.makedirs(expDir, exist_ok=True)
os.makedirs(lossDir, exist_ok=True)
print(baseDir,expDir)



if __name__=="__main__":

    
    batch_size = 8
    train_set = MusicDataset(x_tr,y_tr)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    val_set = MusicDataset(x_val,y_val)
    validationloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    

    # Net = Wavenet.Wavenet(len(unique_notes),blocks=4,layers = 2)
    Net = LSTM.Sequence(len(unique_notes),32)
    Net.to(device)     


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    loss_history = []
    best_val_acc = 0


    ''' TRAINING THE MODEL '''
    correct_preds = 0
    total_preds = 0
    for epoch in range(50): 
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            input , label = data
            optimizer.zero_grad()
            output = Net(input.to(device),input.shape[0])
    #         print('here')
            loss = criterion(output.to(device), label.to(device))
            loss.backward()
            optimizer.step()
            total_preds += input.shape[0]
            correct_preds += torch.sum(torch.argmax(output,1) == label.to(device))

            running_loss += loss.item()
            if (i % 300 == 299  ):
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 300))
                loss_history.append(running_loss/300)
                running_loss = 0.0
        train_acc =  correct_preds/total_preds *100 
        trainreport ="Training Accuracy : \n correct predictions  : {} \n total predictions : {} \n Accuracy : {} \n ------------------------\n".format(correct_preds,total_preds,train_acc)
        print(trainreport)     
        correct_preds = 0
        total_preds = 0
        val_loss = 0
        for i, data in enumerate(validationloader, 0):
            input , label = data
            output = Net(input.to(device),input.shape[0])
            loss = criterion(output.to(device), label.to(device))
            val_loss += loss.item()
            total_preds += input.shape[0]
            correct_preds += torch.sum(torch.argmax(output,1) == label.to(device))
        val_acc = correct_preds/total_preds *100
        val_loss /= total_preds
        valreport ="Validation Accuracy : \n correct predictions  : {} \n total predictions : {} \n Val-Loss : {} \n ------------------------\n".format(correct_preds,total_preds,val_acc )
        print(valreport)
        if(val_acc > best_val_acc):
            print('Updating the saved model')
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": Net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_acc": train_acc,
                "valid_acc": val_acc
            }
            torch.save(checkpoint, os.path.join(expDir, 'checkpoint.tar'))
            torch.save(Net, os.path.join(expDir, 'model.pth'))
        
        correct_preds = 0
        total_preds = 0


    print('Finished Training')

    plt.figure()
    plt.plot(loss_history)
    plt.savefig(lossDir + '/loss_history.png')

