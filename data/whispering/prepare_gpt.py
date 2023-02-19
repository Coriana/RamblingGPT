import os
import requests
import numpy as np
from tqdm import tqdm
import concurrent.futures
import pickle
import random
import tiktoken

tdata = []
vdata = []

def remove_caseifer(text):
    new_text = ""
    i = 0
    while i < len(text):
        if text[i] == "^":
            if i+1 < len(text):
                new_text += text[i+1].upper()
                i += 1
            else:
                pass  # skip this index
        else:
            new_text += text[i]
        i += 1
    return new_text
    
def add_caseifer(text):
    tokenlist = set("\n\" !$&'#,=-<>*@.:;[]?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    upperlist = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    new_text = ""
    for char in text:
        if char in tokenlist:
            if char in upperlist:
                new_text += "^" + char.lower()
            else:
                new_text += char
        else:
            continue
            
    return new_text

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        #with open("input.txt", "a") as ifile:
        #    ifile.write(data)
            
        # data = add_caseifer(data)
       # first_subdir = file_path.split(os.sep)[-2]
        if np.random.rand() < 0.975:
            tdata.append(data)
        else:
            vdata.append(data)

                # print(file_path)

    except:
        print(f"File {file_path} failed to process")
        
root_dir = r'D:\ML\whispering'
print("loading file list")
file_paths = []
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if file_path.endswith(".txt"):
            file_paths.append(file_path)
            
random.shuffle(file_paths)

print("File list created beginning processeing now.")
#train_file = open(os.path.join(os.path.dirname(__file__), 'train.bin'), 'ab')
#val_file = open(os.path.join(os.path.dirname(__file__), 'val.bin'), 'ab')

for file_path in tqdm(file_paths):
    process_file(file_path)
    

print(f"Loaded files")

#print(f"length of dataset in characters: {len(data):,}")
#data = traindata + valdata
# get all the unique characters that occur in this text
# encode with tiktoken gpt2 bpe

enc = tiktoken.get_encoding("gpt2")
# create the train and test splits
#n = len(data)
#train_data = data[:int(n*0.9)]
#val_data = data[int(n*0.9):]

# encode both to integers
#train_ids = encode(traindata)
#val_ids = encode(valdata)
#print(f"train has {len(train_ids):,} tokens")
#print(f"val has {len(val_ids):,} tokens")

# export to bin files
val_ids = enc.encode_ordinary(''.join([str(elem) for elem in vdata]))
print(f"val has {len(val_ids):,} tokens")
val_ids = np.array(val_ids, dtype=np.uint16)
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'gpt_val.bin'))
val_ids = ""

train_ids = enc.encode_ordinary(''.join([str(elem) for elem in tdata]))
print(f"train has {len(train_ids):,} tokens")
train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'gpt_train.bin'))



