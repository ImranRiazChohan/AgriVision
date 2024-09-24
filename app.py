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

def precaution_convert_into_arabic(api,parameters):
    response=requests.post(api,json=parameters)
    if response.status_code == 200:
        return response.json()["translation"]
    else:
        print(response.status_code)
    
os.environ["GROQ_API_KEY"]="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"

model= model = YOLO('best.pt')

st.sidebar.title("AgriVision")

on=st.sidebar.toggle("Arabic")

options=st.sidebar.radio("File Upload",("Upload Image","Live Camera","chatbot"))
if options=="Upload Image":

    image=st.file_uploader("",type=['jpeg','png','jpg'])
    if image is not None:
        st.image(image,use_column_width=True)
        uploaded_image = Image.open(image)

        results=model.predict(uploaded_image,conf=0.5)
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

    