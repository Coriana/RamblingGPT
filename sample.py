"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import time
import random
import pyttsx3
import concurrent.futures
import sys
from time import sleep
import socket

def send_data(data, port):
    s = socket.socket()
    s.connect(('localhost', port))
    s.send(data.encode())
    s.close()
    
def gentext(history):
    start_ids = encode(add_caseifer(history))
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    generated_data = ''
    for k in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        output = remove_caseifer(decode(y[0].tolist()))[len(history):]
        generated_data = output.split("\n", 1)[0]
        if generated_data == '\n':
            generated_data = output.split("\n", 2)[0]
            if generated_data == '\n':
                generated_data = output.split("\n", 3)[0]
       # while not generated_data.strip():  # loop until parsed_output contains non-empty characters
       #     generated_data = generated_data[generated_data.find("\n") + 1:]  # find the first newline character and remove it from the parsed_output

        generated_sequences.append(generated_data)
    
    #for i, sequence in enumerate(generated_sequences):
        #print(f"Output {i}: {sequence}")    
    
    #selected_output = int(input("Select output number to use: "))         
    selected_output = random.randrange(num_samples) 
    actual_output = generated_sequences[selected_output]
    return actual_output

def typing(text):
    for char in text:
        sleep(0.02)
        sys.stdout.write(char)
        sys.stdout.flush()
    sys.stdout.write('\n')
    
        
def textToSpeech(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id) # female voice
    engine.setProperty('rate', 175) # change the speaking rate
    engine.say(text)
    engine.runAndWait()
    del engine

def parallel(text):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_tasks = {executor.submit(textToSpeech, text), executor.submit(typing, text)}
        for future in concurrent.futures.as_completed(future_tasks):
            try:
                data = future.result()
            except Exception as e:
                print(e)
                
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
    new_text = ""
    for char in text:
        if char.isupper():
            new_text += "^" + char.lower()
        else:
            new_text += char
    return new_text
    
#def Read_input():
    # do this

#def ContinueOutput():
    # do that

#def StartFromNothing():
    # do the other thing
# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'lamda' # ignored if init_from is not 'resume'
start = "Hello.\nHow are you?\nI am doing well, how about yourself?\nYou know another day raiding.\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 4 # number of samples to draw
max_new_tokens = 150 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1336
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
        
        
start_ids = encode(add_caseifer(start))
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
history = start
lastsaid = start
generated_sequences = []

# run generation
with torch.no_grad():
    with ctx:
        while True:
            #text = input()  # wait for user input
            #history = history + text + '\n' # append input to history
            start_ids = encode(add_caseifer(history))
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            generated_data = ''
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                output = remove_caseifer(decode(y[0].tolist()))[len(history):]
                generated_data = output.split("\n", 1)[0]
                if generated_data == '\n':
                    generated_data = output.split("\n", 1)[1]
                    if generated_data == '\n':
                        generated_data = output.split("\n", 1)[2]
               # while not generated_data.strip():  # loop until parsed_output contains non-empty characters
               #     generated_data = generated_data[generated_data.find("\n") + 1:]  # find the first newline character and remove it from the parsed_output
                while generated_data == lastsaid:
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    output = remove_caseifer(decode(y[0].tolist()))[len(history):]
                    generated_data = output.split("\n", 1)[0]
                    if generated_data == '\n':
                        generated_data = output.split("\n", 1)[1]
                        if generated_data == '\n':
                            generated_data = output.split("\n", 1)[2]
                            
                generated_sequences.append(generated_data)
            
            #for i, sequence in enumerate(generated_sequences):
                #print(f"Output {i}: {sequence}")    
            
            #selected_output = int(input("Select output number to use: "))         
            selected_output = random.randrange(num_samples) 
            actual_output = generated_sequences[selected_output]

            history = history + actual_output + '\n'
            # send_data(actual_output, 1234)
            typing(actual_output)
            lastsaid = actual_output
           # if actual_output.endswith('.'):
           #     sleep_time = random.uniform(0.25, 2)
           #     time.sleep(sleep_time)
           # elif actual_output.endswith('?'):
           #     sleep_time = random.uniform(0.5, 3)
           #     time.sleep(sleep_time)
            #parallel(actual_output)
            generated_sequences = []

            #print('---------------')


