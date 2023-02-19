import os
import requests
import numpy as np
from tqdm import tqdm
import concurrent.futures
import pickle
import random
import hashlib

root_dir = r'C:\Dev\data\whispering'

train_file_path = os.path.join(os.path.dirname(__file__), 'train.txt')
val_file_path = os.path.join(os.path.dirname(__file__), 'val.txt')
tdata = []
vdata = []
hash_dir = r'hashes'
baddir = r'fails'

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
    uppers = 0
    lowers = 0
    tokenlist = set("\n\" !$&'#,=-<>*@.:;[]()?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    upperlist = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    lowerlist = set("abcdefghijklmnopqrstuvwxyz")
    new_text = ""
    for char in text:
        if char in tokenlist:
            if char in upperlist:
                uppers += 1
                new_text += "^" + char.lower()
            elif char in lowerlist:
                lowers += 1
                new_text += char
            else:
                new_text += char
        else:
            continue
            
    return new_text, uppers, lowers

def create_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()

def is_processed(directory, file_hash):
    if os.path.isfile(os.path.join(directory, file_hash)):
        return True
    return False

def is_not_failed(directory, file_hash):
    if os.path.isfile(os.path.join(directory, file_hash)):
        return False
    return True

def is_not_processed(directory, file_hash):
    if os.path.isfile(os.path.join(directory, file_hash)):
        return False
    return True

def process_file(file_path):
    try:
        file_hash = create_hash(file_path)
    except:
        print(f"{file_path} failed to hash")
    if is_not_processed(hash_dir, file_hash) and is_not_failed(baddir, file_hash):
        try:
            with open(file_path, 'r', encoding='ansi') as f:
                data = f.read()
            #with open("input.txt", "a") as ifile:
            #    ifile.write(data)
                
            data, ups, lows = add_caseifer(data)

            if lows < ups: 
                with open(os.path.join(baddir, file_hash), "w") as f:
                    f.write("")
                # print(f"{file_path} has too many uppercase {ups} vs {lows}")
                return
        except:
            print(f"File {file_path} failed to process to format")

        try:
            if np.random.rand() < 0.985:
                tdata.append(data)
                try:
                    with open(os.path.join(hash_dir, file_hash), "w") as f:
                        f.write("")
                except:
                    print(f"File {file_path} failed to process to traindata")

            else:
                vdata.append(data)
                try:
                    with open(os.path.join(hash_dir, file_hash), "w") as f:
                        f.write("")
                except:
                    print(f"File {file_path} failed to process to valdata")

        except:
            print(f"File {file_path} failed to process to data")

                    # print(file_path)
            
            
chars = "\n\" !$&'#,=-<>*@.:;[]()?^0123456789abcdefghijklmnopqrstuvwxyz"

if not os.path.exists(hash_dir):
    os.makedirs(hash_dir)
    
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

#valdata = add_caseifer(valdata)

#print(f"processed Validate to casified")
#valdata = ''.join([str(elem) for elem in vdata])
#traindata = ''.join([str(elem) for elem in tdata])
#traindata = add_caseifer(traindata)
#print(f"processed train to casified")
#print(f"length of dataset in characters: {len(data):,}")
#data = valdata + traindata
# get all the unique characters that occur in this text
#chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")
#chars = "\n !$&',-.:;?^0123456789abcdefghijklmnopqrstuvwxyz"

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
#n = len(data)
#train_data = data[:int(n*0.9)]
#val_data = data[int(n*0.9):]

# encode both to integers
val_ids = encode(''.join([str(elem) for elem in vdata]))
print(f"val has {len(val_ids):,} tokens")
# export to bin files
val_ids = np.array(val_ids, dtype=np.uint16)
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
val_ids = ""

train_ids = encode(''.join([str(elem) for elem in tdata]))
print(f"train has {len(train_ids):,} tokens")
train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)


