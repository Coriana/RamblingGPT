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
import re
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('samples.db')
c = conn.cursor()

# Create a table to store the samples
c.execute('''CREATE TABLE IF NOT EXISTS samples
             (history text, chosen_output text, rejected_sample text)''')
MAX_HISTORY_LENGTH = 360

class History:
    def __init__(self):
        self.lines = []
        self.length = 0
        self.name = ""
        self.direction = ""
    
    def add(self, line):
        line_length = len(line)
        while (self.length + line_length) > (MAX_HISTORY_LENGTH - len(self.direction)):
            self.length -= len(self.lines.pop(0))
        self.lines.append(line)
        self.length += line_length
    
    def __str__(self):
        history = self.direction + "\n" + "".join(self.lines)
        if len(history) > MAX_HISTORY_LENGTH:
            history = history[-MAX_HISTORY_LENGTH:]
        return history
        
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
        sleep(0.001)
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
start = "Wecome to the chat. We are just getting started, so come on in and enjoy the show." # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 # number of samples to draw
max_new_tokens = 175 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1336
MAX_LENGTH = 1024
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
direction_file = "direction.txt"
follower_file = "follower.txt"
input_file = "input.txt"
autorun_file = "autoplay.txt"
sample_file = "sampler.txt"

#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
past_history = History()

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
    meta_path = os.path.join(out_dir, 'meta.pkl')
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
        

MAX_LASTHISTORY_LENGTH = 3

# Initialize the last 10 strings said to empty strings
last_said = [''] * MAX_LASTHISTORY_LENGTH
dontsay = ['']
dontsay.append('.')
forbidden_words = ''
with open("badwords.txt", "r") as f:
    forbidden_words = [line.strip() for line in f.readlines()]
    
badwords = [re.compile(rf"\b{word}\b", re.IGNORECASE) for word in forbidden_words]

start_ids = encode(add_caseifer(start))
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
history = start
#lastsaid = start
generated_sequences = []
#past_history.name = "I am Emeldar and "
past_history.direction = "Eml: I'm Emelda."
past_history.add(start+ '\n')
print(str(past_history))
# run generation
GEN = True
wasinputted = False
with torch.no_grad():
    with ctx:
        while True:
            if os.path.isfile(sample_file) and os.path.getsize(sample_file) > 0:
                with open(sample_file, "r") as f:
                    txt = f.read().strip() 
                    num_samples = int(txt)
                    
            if os.path.isfile(direction_file) and os.path.getsize(direction_file) > 0:
                with open(direction_file, "r") as f:
                    past_history.direction = f.read()
                with open(direction_file, "w") as f:
                    f.writelines('')
               # GEN = True
                    
            if os.path.isfile(follower_file) and os.path.getsize(follower_file) > 0:
                with open(follower_file, "r") as f:
                     lines = f.readlines()

                # Select a random line from the file and remove it
                #random_line = random.choice(lines)
                #lines.remove(random_line)
                inputted_data = lines[0]
               # clean = re.sub(r"[\n\r\t\v\f]", " ", inputted_data)
                lines.pop(0)
                with open(follower_file, "w") as f:
                    f.writelines(lines)
                if not any(pattern.search(inputted_data) for pattern in badwords):
                    past_history.add(inputted_data)
                    wasinputted = True

                    GEN = True                      
                    
            if os.path.isfile(input_file) and os.path.getsize(input_file) > 0 and GEN == False:
                with open(input_file, "r") as f:
                     lines = f.readlines()

                # Select a random line from the file and remove it
                #random_line = random.choice(lines)
                #lines.remove(random_line)
                inputted_data = lines[0]
               # clean = re.sub(r"[\n\r\t\v\f]", " ", inputted_data)
                lines.pop(0)
                with open(input_file, "w") as f:
                    f.writelines(lines)
                if not any(pattern.search(inputted_data) for pattern in badwords):
                    past_history.add(inputted_data)
                    wasinputted = True
                    GEN = True                      
                
            if random.randint(1, 10) == 5:
                GEN = True
            else:
                time.sleep(0.5)
                
                
            if os.path.isfile(autorun_file) and os.path.getsize(autorun_file) > 0:
                GEN = True
                
            if GEN:
                history = str(past_history) + 'Eml: ' # append input to history
                try:
                    start_ids = encode(add_caseifer(history))
                except:
                    continue
                # print(history)
                x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
                generated_data = ''
                for k in range(num_samples):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    output = remove_caseifer(decode(y[0].tolist()))[len(history):]
                    generated_data = output.split("\n", 1)[0]

                    while any(generated_data == s for s in last_said) or any(generated_data == s for s in dontsay) or any(pattern.search(generated_data) for pattern in badwords):
                        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                        output = remove_caseifer(decode(y[0].tolist()))[len(history):]
                        generated_data = output.split("\n", 1)[0]
                                    
                    generated_sequences.append(generated_data)
                
                if (os.path.isfile(sample_file) and os.path.getsize(sample_file) > 0) and num_samples > 1:
                    for i, sequence in enumerate(generated_sequences):
                        print(f"Output {i}: {sequence}")    
                    selected_output = int(input("Select output number to use: "))        
                    actual_output, rejected_samples = generated_sequences[selected_output]

                    # Insert the data into the samples table
                    for rejected_sample in rejected_samples:
                        c.execute("INSERT INTO samples VALUES (?, ?, ?)", (history, actual_output, rejected_sample))

                    c.execute("INSERT INTO samples VALUES (?, ?, ?)", (history, actual_output, ''))

                    # Commit the changes and close the connection
                    conn.commit()
                    conn.close()                    
                else:
                    selected_output = random.randrange(num_samples) 
                    actual_output = generated_sequences[selected_output]
                past_history.add(actual_output + '\n')
              #  past_history.add(actual_output)
                #history = history + actual_output + '\n'
                #history = history[:MAX_LENGTH]
                if wasinputted:
                    try:
                        send_data(inputted_data + actual_output, 1234)
                    except:
                        failed = 1
                    typing(inputted_data + actual_output)
                    wasinputted = False
                    inputted_data = ''
                else:
                    try:
                        send_data(actual_output, 1234)
                    except:
                        failed = 1
                    typing(actual_output)
                #print(actual_output)
                last_said.append(actual_output)
                last_said.pop(0)
                #if actual_output.endswith('.'):
                #    sleep_time = random.uniform(0.25, 2)
                #    time.sleep(sleep_time)
                #elif actual_output.endswith('?'):
                #    sleep_time = random.uniform(0.5, 3)
                #    time.sleep(sleep_time)
                #parallel(actual_output)
                generated_sequences = []
                GEN = False


