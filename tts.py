import socket
import pyttsx3
import random
import select
import sys
import os

def speak(data):
    voicespeed = random.randrange(170, 185) 
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id) # female voice
    engine.setProperty('rate', voicespeed) # change the speaking rate
    engine.say(data)
    engine.runAndWait()

def start_server(port):
    s = socket.socket()
    s.bind(('localhost', port))
    s.listen(1)
    print(f"Server is running on port {port}...")
    speech_on = True

    while True:
        conn, addr = s.accept()
        # print(f"Connection from {addr[0]}:{addr[1]}")
        data = conn.recv(1024).decode()
        # print(f"Received data: {data}")
        if os.path.isfile('tts_off.txt') and os.path.getsize('tts_off.txt') > 0:
            speech_on = True
        else:
            speech_on = False
        print(data)
        if speech_on:
            speak(data)
            

    conn.close()

if __name__ == "__main__":
    start_server(1234)
