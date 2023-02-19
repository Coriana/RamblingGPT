import os
import requests
import numpy as np
from tqdm import tqdm
import concurrent.futures
import pickle
import random
import tiktoken
import hashlib

train_file_path = os.path.join(os.path.dirname(__file__), 'train.txt')
val_file_path = os.path.join(os.path.dirname(__file__), 'val.txt')
train_hash_dir = r'trainhashes'
val_hash_dir = r'valhashes'

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
def cleandata(text):
    tokenlist = set("\n\" !$&',-.:;?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    new_text = ""
    for char in text:
        if char in tokenlist:
            new_text += char
        else:
            continue
            
    return new_text

def create_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()

def is_processed(hash_dir, file_hash):
    if os.path.isfile(os.path.join(hash_dir, file_hash)):
        return True
    return False

def process_file(file_path):
    try:
        file_hash = create_hash(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        #with open("input.txt", "a") as ifile:
        #    ifile.write(data)
            
        data = add_caseifer(data)
        #data = cleandata(data)
        if is_processed(train_hash_dir, file_hash):
            with open(train_file_path, "a", encoding='utf-8') as tfile:
                tfile.write(data)            
        elif is_processed(val_hash_dir, file_hash):
            with open(val_file_path, "a", encoding='utf-8') as vfile:
                vfile.write(data)            

        else:
            if np.random.rand() < 0.98:
                with open(train_file_path, "a", encoding='utf-8') as tfile:
                    tfile.write(data)            
                    with open(os.path.join(train_hash_dir, file_hash), "w") as f:
                        f.write("")
            else:
                with open(val_file_path, "a", encoding='utf-8') as vfile:
                    vfile.write(data)
                    with open(os.path.join(val_hash_dir, file_hash), "w") as f:
                        f.write("")

                # print(file_path)
                

    except:
        print(f"File {file_path} failed to process")
        
chars = "\n\" !$&'#,=-<>*@.:;[]?^0123456789abcdefghijklmnopqrstuvwxyz"


with open(train_file_path, "w", encoding='utf-8') as f:
    f.write("")
with open(val_file_path, "w", encoding='utf-8') as f:
    f.write("")

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
    
