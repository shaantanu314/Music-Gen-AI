import numpy as np
from music21 import *
from collections import Counter

def get_sequences(notes_array,timesteps=32,future_steps=8):

    notes_ = [element for note_ in notes_array for element in note_]
    freq = dict(Counter(notes_))
    frequent_notes = [note_ for note_, count in freq.items() if count>=50]
    
    
    new_music=[]

    for notes in notes_array:
        temp=[]
        for note_ in notes:
            if note_ in frequent_notes:
                temp.append(note_)            
        new_music.append(temp)
        
    new_music = np.array(new_music,dtype=object)


    no_of_timesteps = timesteps
    x = []
    y = []
    for note_ in new_music:
        for i in range(0, len(note_) - (no_of_timesteps+future_steps), 1): 
            #preparing input and output sequences
            input_ = note_[i:i + no_of_timesteps]
            output = note_[i+1+no_of_timesteps:i+1+no_of_timesteps+future_steps]
            
            x.append(input_)
            y.append(output)
            
    x=np.array(x)
    y=np.array(y)

    #preparing input sequence
    unique_notes = list(set(x.ravel()))
    note_to_int = dict((note_, number) for number, note_ in enumerate(unique_notes))
    x_seq=[]
    for i in x:
        temp=[]
        for j in i:
            #assigning unique integer to every note
            temp.append(note_to_int[j])
        x_seq.append(temp)
        
    x_seq = np.array(x_seq)

    y_seq=[]
    for i in y:
        temp=[]
        for j in i:
            #assigning unique integer to every note
            temp.append(note_to_int[j])
        y_seq.append(temp)
        
    y_seq = np.array(y_seq)


    return x_seq,y_seq,unique_notes,note_to_int