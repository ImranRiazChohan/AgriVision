import streamlit as st
import cv2
from groq import Groq
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np
import os
import requests

def precaution_bot(disease=""):
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
            "role":"user",
            'content':f"""give me top 3 precaution and prevention for Dates Palm Disease 
            that i have mentioned{disease} in bullit point as in english 
            this is for farmer so please easy and understanable"""
            }
            ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    precaution=''
    for chunk in completion:
        precaution+= chunk.choices[0].delta.content or ""
    return precaution


def precaution_chatbot(prompt=""):
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
            "role":"user",
            'content':f"""{prompt}"""
            }
            ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    precaution=''
    for chunk in completion:
        precaution+= chunk.choices[0].delta.content or ""
    return precaution

def precaution_convert_into_arabic(api,parameters):
    response=requests.post(api,json=parameters)
    if response.status_code == 200:
        return response.json()["translation"]
    else:
        print(response.status_code)
    
os.environ["GROQ_API_KEY"]="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"

model= model = YOLO('best.pt')
st.logo("logo.svg")

st.sidebar.title("AgriVision")

on=st.sidebar.toggle("Arabic")

options=st.sidebar.radio("File Upload",("Upload Image","Live Camera","AI-Agent"))
if options=="Upload Image":
    image=st.file_uploader("",type=['jpeg','png','jpg'])
    if image is not None:
         # Read the uploaded file as an OpenCV image
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb,channels="RGB")

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.image(gray_image) 

        results=model.predict(image_rgb,conf=0.5)
        result = results[0]
        class_names = result.names
        probs = result.probs.data.tolist()
        st.write(class_names[np.argmax(probs)])

        st.title("Precautions!")
        precaution=precaution_bot(class_names[np.argmax(probs)])
        with st.status("", expanded=True) as status:
            if on:
                text=precaution_convert_into_arabic("https://deep-translator-api.azurewebsites.net/google/",{"source": "english","target": "arabic","text": f"{precaution}","proxies": []})
                st.write(text)
            else:
                st.write(precaution)

elif options=="Live Camera":
    picture=st.camera_input("Take a Picture")

elif options=="AI-Agent":
    # Set the title of the app
    st.title("Palm-Disease bot ")

    # Initialize the chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Chat input
    prompt = st.chat_input("Say something")


    if prompt:
        response=precaution_chatbot(prompt)
        # Store the user input
        st.session_state.messages.append({"role": "user", "content": prompt})

        # # Simple bot response
        # response = f"{prompt}"
        st.session_state.messages.append({"role": "bot", "content": response})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Display bot response
        with st.chat_message("assistant"):
            st.write(response)

   
    # Display the chat history
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
    