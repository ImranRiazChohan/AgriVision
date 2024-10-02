<br />
<div align="center">
    <img src="assets/AgriVision.png" alt="Logo" />
    <h3 align="center">🤖 AI-DRIVEN INNOVATIONS FOR SUSTAINABLE DATE FARMING 🌴 </h3>
    <p align="center">
    An Smart Solutions for Disease Prevention and Crop Health Optimization
    <br />
  </p>
</div>
</div>
<br>

## Problem Statement
Despite advancements in agricultural technology, farmers face significant challenges in the early detection and management of plant diseases, leading to substantial crop losses. Current methods often struggle with issues such as noise reduction, effective feature extraction, and redundant data elimination, resulting in inefficient disease identification and delayed interventions. Moreover, there is a pressing need for deep learning models that are not only effective but also efficient, with fewer parameters and faster training times. This research seeks to address these challenges by integrating advanced AI techniques—specifically, **YOLOv8** for disease detection and **LLama3** for preventive recommendations—aimed at revolutionizing disease management in Palm date farming and ensuring a sustainable agricultural future.

## Objective
The primary objective of this research is to develop and implement advanced deep learning models for the early detection, classification, and prevention of Palm leaf diseases, utilizing **YOLOv8** for precise disease detection and **LLama3** for decision support. The goal is to enhance agricultural practices by providing farmers with real-time recommendations and automated solutions that improve crop management efficiency, reduce crop losses, and promote sustainable date farming.

## Features

### Real-Time Image Segmentation for Disease Detection and Prevention

**Description:**  
Develop a deep learning model that utilizes image segmentation techniques to identify and classify Palm leaf diseases from images. The model will analyze segmented areas of the leaf to provide detailed insights into the disease's extent and severity.

**Features:**
- **Image Input Analysis:** Users can upload images of Palm leaves, which the model will segment to highlight affected areas.
- **Prevention and Precaution Recommendations:** Based on the disease detected, the system generates tailored recommendations for prevention and treatment, including appropriate fungicides, watering techniques, and environmental adjustments.
- **User-Friendly Interface:** An easy-to-navigate platform for farmers to access disease information and recommendations.

### AI-Powered Chatbot for Decision Support

**Description:**  
Develop an AI-driven chatbot that farmers can interact with to seek advice on Palm leaf disease management. This chatbot will be powered by the LLama3 model, offering conversational support based on user queries.

**Features:**
- **Interactive Q&A:** Farmers can ask questions related to disease identification, treatment options, and best practices for prevention.


## Palm Tree Disease Dataset

### Overview
This repository contains a dataset focused on palm tree diseases, which has been annotated for use in machine learning applications. The dataset aims to support research and development in the identification and treatment of diseases affecting palm trees.

### Original Dataset
The original dataset was sourced from [dataset](https://drive.google.com/drive/folders/1GsEGFi5REM1Lc0185RKKpVc9CjqOf68D?usp=sharing). It includes a wide range of images showcasing various conditions affecting palm trees.

## Dataset Description
The palm tree disease dataset includes images of various palm trees affected by different diseases. Each image has been annotated to highlight the symptoms, allowing for better training of models that can identify these diseases in the wild.

### Key Features
- **Image Count:** 460 
- **Disease Types:** **Black_Scorch**,**Parlatoria_Blanchardi**
- **Annotation Tool:** The dataset has been annotated using [RoboFlow](https://roboflow.com), a powerful tool for creating and managing image datasets.

### Data Access
You can access and download the annotated dataset from the following link:

[Download Palm Tree Disease Dataset](https://universe.roboflow.com/imran-riaz-chohan/dates_leaves_disease_seg_dataset/dataset/1)

### Screenshots
Below are some screenshots illustrating the annotations done using RoboFlow:

![Screenshot 1](https://github.com/ImranRiazChohan/Medina_Hackathon_Dates/blob/main/assets/class_1.PNG)  
*Example of a palm tree affected by **Black-Scorc**h*

![Screenshot 2](https://github.com/ImranRiazChohan/Medina_Hackathon_Dates/blob/main/assets/class_2.PNG)  
*Example of a palm trees affected by **Parlatoria_Blanchardi***

![data_annotation](https://github.com/user-attachments/assets/6c70173d-b8a7-480a-b9f6-7e8b1a1909fa)



## Preparing a custom dataset on Roboflow

Building a custom dataset can be a painful process. It might take dozens or even hundreds of hours to collect images, label them, and export them in the proper format. Fortunately, Roboflow makes this process as straightforward and fast as possible. Let me show you how!

### Step 1: Creating project

Before you start, you need to create a Roboflow [account](https://app.roboflow.com/login). Once you do that, you can create a new project in the Roboflow [dashboard](https://app.roboflow.com/). Keep in mind to choose the right project type. In our case, Object Detection.

<div align="center">
  <img
    width="640"
    src=""
  >
</div>

### Step 2: Uploading images

Next, add the data to your newly created project. You can do it via API or through our [web interface](https://docs.roboflow.com/adding-data/object-detection).

If you drag and drop a directory with a dataset in a supported format, the Roboflow dashboard will automatically read the images and annotations together.

<div align="center">
  <img
    width="640"
    src="https://ik.imagekit.io/roboflow/preparing-custom-dataset-example/uploading-images.gif?ik-sdk-version=javascript-1.4.3&updatedAt=1672929808290"
  >
</div>

### Step 3: Generate new dataset version

Now that we have our images and annotations added, we can Generate a Dataset Version. When Generating a Version, you may elect to add preprocessing and augmentations. This step is completely optional, however, it can allow you to significantly improve the robustness of your model.

<div align="center">
  <img
    width="640"
    src="https://media.roboflow.com/preparing-custom-dataset-example/generate-new-version.gif?ik-sdk-version=javascript-1.4.3&updatedAt=1673003597834"
  >
</div>

### Step 4: Exporting dataset

Once the dataset version is generated, we have a hosted dataset we can load directly into our notebook for easy training. Click `Export` and select the `YOLOv8` dataset format.

<div align="center">
  <img
    width="640"
    src="https://ik.imagekit.io/roboflow/preparing-custom-dataset-example/export.gif?ik-sdk-version=javascript-1.4.3&updatedAt=1672943313709"
  >
</div>


## Getting Started

### Prerequisites

* Install Python version 3.11 on the system
  
### Steps to run the project

Open command prompt in the project's folder
1. Clone repository 
  ```sh
  git clone https://github.com/
  ```
2. Goto the project's folder
  ```sh
  cd date_palm_diseases
  ```
3. Install required packages 
  ```sh
  pip install -r requirements.txt
  ```
4. Run project 
  ```sh
  Streamlit run app_v1.py
  ```
## Snapshot
### *Main Page*
![main_page](https://github.com/ImranRiazChohan/Medina_Hackathon_Dates/blob/main/assets/main_page.PNG)
### *Chatbot Page*
![chat_page](https://github.com/ImranRiazChohan/Medina_Hackathon_Dates/blob/main/assets/chatbot_page.PNG)
### *Disease Detection and Prevention Recommendation Page*
![input_1](https://github.com/ImranRiazChohan/Medina_Hackathon_Dates/blob/main/assets/image_segmentation_page_1.PNG)
![input_2](https://github.com/ImranRiazChohan/Medina_Hackathon_Dates/blob/main/assets/image_segmentation_page_2.PNG)
![input_3](https://github.com/ImranRiazChohan/Medina_Hackathon_Dates/blob/main/assets/image_segmentation_page_3.PNG)
## Demo

https://github.com/user-attachments/assets/d299fb32-171f-4229-a73c-71c50c1bb3ec    

## Model Train Notebook Video

https://github.com/user-attachments/assets/1eb4a3a3-e0b2-4633-b4a2-919008eec58e

## Model Results
*Confusion Matrix & Result*

![confusion_matrix (3)](https://github.com/user-attachments/assets/2f6c3d62-4e04-4a1b-aa39-9bd3c8e1e15a)

![results (1)](https://github.com/user-attachments/assets/bedfb40b-da60-4ca3-b8b1-43a6b0b00dfc)


## Languages & Tools
This section shows the frameworks and libraries utilized in the project. 
* Python
* Streamlit
* Opencv
* Llama3
* Ultralytics
* YoloV8

<p align="left"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="50" height="50"/>
<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="streamlit" width="50" height="50"/>
<img src="https://upload.wikimedia.org/wikipedia/commons/5/53/OpenCV_Logo_with_text.png" alt="opencv" width="50" height="50"/>
<img src="https://raw.githubusercontent.com/RMNCLDYO/groq-ai-toolkit/main/.github/groq-logo.png" alt="groq" width="50" height="50"/>
<img src="https://avatars.githubusercontent.com/u/897180?v=4" alt="sk=image" width="50" height="50"/>
</p>



