from music21 import *
import os 
import numpy as np
from midi_helper import *
import Generate_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

train_size = 0.7
validation_size = 0.2 
test_size = 0.1

path='../schubert/'

#read all the filenames
files=[i for i in os.listdir(path) if i.endswith(".mid")]
notes_array = np.array([read_midi(path+i) for i in files],dtype=object)
# notes_array = np.array(read_midi(path+files[1]))

x_seq , y_seq , unique_notes,note_to_int  = Generate_dataset.get_sequences(notes_array,timesteps=32,future_steps=8)

X_train, X_rem, y_train, y_rem = train_test_split(x_seq,y_seq, train_size=train_size)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=float(test_size)/float(validation_size+test_size))

train_dataset = [{'x_tr':X_train[i],'future':y_train[i]} for i in range(0,len(X_train))]
validation_dataset = [{'x_val':X_valid[i],'future':y_valid[i]} for i in range(0,len(X_valid))]
test_dataset = [{'x_test':X_test[i],'future':y_test[i]} for i in range(0,len(X_test))]
# print(len(train_dataset),len(validation_dataset),len(test_dataset))

df_tr = pd.DataFrame(train_dataset)
df_val = pd.DataFrame(validation_dataset)
df_test = pd.DataFrame(test_dataset)
df_notes = pd.DataFrame(unique_notes)

df_tr.to_csv('trainset.csv')
df_val.to_csv('validationset.csv')
df_test.to_csv('testset.csv')
df_notes.to_csv('notes.csv')
# print(unique_notes)
