import streamlit as st
import cv2
from groq import Groq
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np
import os
import requests
from skimage.measure import shannon_entropy
from skimage import measure

def precaution_bot(disease, num_detections, detection_details):
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages = [
                    {
                        'role': 'system',
                        'content': (
                            'You are a knowledgeable botanist. Your role is to provide detailed plant care precautions and recommend appropriate treatments or medicines for various plant issues. '
                            'Ensure your advice is clear, practical, and based on best practices in botany. '
                            'The user will provide the name of the disease and the detection results of an affected plant. '
                            'Based on this information, you should give a response that includes appropriate care precautions and treatment recommendations, incorporating the detection results directly into the advice.'
                        )
                    },
                    {
                        'role': 'user',
                        'content': (
                            'give me top 3 precaution and prevention for Dates Palm Disease '
                            f'that I have detected {disease} disease with {num_detections} number of defected areas and contain {detection_details} in my image. '
                            'Write the bullet points in English because this is for a farmer, so make it easy and understandable.'
                        )
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

model=  YOLO('best_seg.pt')


st.set_page_config(
    page_title='AgriVision'
)

st.image("banner_hd.png",use_column_width=True)

body='''<h3>Professor Dr. Tanzila Saba</h3>
<h5>Research Professor / Lab Leader 
Associate Director,Research And Initiative Center(RIC)</h5>
'''
st.markdown(body, unsafe_allow_html=True)
st.divider()
st.logo("AgriVision.png")
st.sidebar.image("AgriVision.png")


on=st.sidebar.toggle("Arabic")
options=st.sidebar.radio("File Upload",("Upload Image","Live Camera","AI-Agent"))
if options=="Upload Image":
    st.title("Upload Palm Tree Leave Images")
    image=st.file_uploader("",type=['jpeg','png','jpg'])
    if image is not None:
         # Read the uploaded file as an OpenCV image
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_width_pixels = 300
        image_height_pixels = 300
        # Define the pixel density (DPI)
        dpi = 96
         # Define the desired size
        desired_size = (image_width_pixels, image_height_pixels)  # Width, Height


        # Convert pixels to centimeters
        image_width_cm = (image_width_pixels / dpi) * 2.54
        image_height_cm = (image_height_pixels / dpi) * 2.54
        # Resize both images
        image_rgb_resized = cv2.resize(image_rgb, desired_size)
        gray_image_resized = cv2.resize(gray_image, desired_size)
        # Binarize the grayscale image
        _, binary_image_resized = cv2.threshold(gray_image_resized, 128, 255, cv2.THRESH_BINARY)

        results = list(model(image_rgb_resized))  # Pass the RGB image to the model
        # Access the first result (if there's only one image processed)
        result = results[0]
        # Get the boxes attribute which contains the confidence scores
        boxes = result.boxes
        class_name = ''
        num_detections = 0
        detection_details = ''
        segmented_areas = []
        total_affected_area = 0
        total_entropy = 0
        mean_area = 0
        rms_area = 0
        total_perimeter = 0

        # Check if there are any boxes
        if boxes and boxes.conf.numel() > 0:
            # Print the number of detections
            num_detections = boxes.conf.numel()
            print(f"Number of detections: {num_detections}")

            # Find the index of the box with the highest confidence
            max_confidence_index = boxes.conf.argmax()

            # Get the class ID of the box with the highest confidence
            class_id = boxes.cls[max_confidence_index].item()

            # Get the class name using the class ID
            class_name = result.names[class_id]

            print(f"Class with highest confidence: {class_name}")

            pixel_area_cm = (image_width_cm / image_width_pixels) * (image_height_cm / image_height_pixels)
            total_image_area_cm = image_width_cm * image_height_cm


            total_entropy = 0

            for i, confidence in enumerate(boxes.conf):
                mask = result.masks.data[i]
                area_in_pixels = mask.sum().item()
                area_in_cm = area_in_pixels * pixel_area_cm
                segmented_areas.append(area_in_cm)
                detection_details += f"Detection {i + 1}: with Confidence = {confidence.item()} and Segmented Area = {area_in_cm} cm² "
                # Calculate entropy for each mask and add to total entropy
                entropy_value = shannon_entropy(mask.cpu().numpy())
                total_entropy += entropy_value
                # Calculate perimeter
                label = measure.label(mask.cpu().numpy())
                props = measure.regionprops(label)
                for prop in props:
                    total_perimeter +=prop.perimeter
                    print(f"Detection {i + 1}: Perimeter = {total_perimeter:.3f}")
            print(detection_details)

            # Calculate total affected area
            total_affected_area = sum(segmented_areas)
            # Calculate mean and RMS of segmented areas
            mean_area = np.mean(segmented_areas)
            rms_area = np.sqrt(np.mean(np.square(segmented_areas)))

            # Round the values to 3 decimal places
            total_affected_area = round(total_affected_area, 3)
            total_entropy = round(total_entropy, 3)
            mean_area = round(mean_area, 3)
            rms_area = round(rms_area, 3)

            print(f"Total Affected Area: {total_affected_area} cm²")
            print(f"Total Entropy Value: {total_entropy}")
            print(f"Total Total Perimeter: {total_perimeter} ")
            print(f"Mean Segmented Area: {mean_area} cm²")
            print(f"RMS Segmented Area: {rms_area} cm²")

        else:
            print("No objects detected.")

        # print('-------- results', results)
        # result = results[0]
        # no_of_masks, x, y = result.masks.shape
        st.divider()
        st.markdown('<h3>Image Analysis</h3>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
       
        # Add the first row of columns for the RGB and grayscale images
        # col1, col2 = st.columns(2)
        col3_diplay = False


        # Display the RGB image in the first column
        with col1:
            st.image(image_rgb_resized, channels="RGB", caption='RGB Image')

        # Display the grayscale image in the second column
        with col2:
            st.image(gray_image_resized, caption='Grayscale Image')

        # Add a second row of columns for the segmented image and binary mask
        # col3, col4 = st.columns(2)

        # Display the segmented image in the third column
        with col3:
            res_plotted = result.plot(line_width=1, labels=True)  # Ensure result is in RGB
            class_caption = f"'{class_name}' disease detected"
            st.image(res_plotted, caption=class_caption, use_column_width=True)

        # Create and display the binary mask in the fourth column
        with col4:
            if boxes and boxes.conf.numel() > 0:
                # Create an empty binary mask
                binary_mask = np.zeros_like(image_rgb_resized[:, :, 0], dtype=np.uint8)

                for i in range(len(boxes.conf)):
                    mask = result.masks.data[i].cpu().numpy()
                    # Resize the mask to match the binary mask dimensions
                    resized_mask = cv2.resize(mask, (binary_mask.shape[1], binary_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    binary_mask[resized_mask > 0] = 255  # Set mask areas to white
                col3_diplay = True
                st.image(binary_mask, caption='Affected Area', use_column_width=True)
                
            else:
                st.write("No objects detected.")
        st.divider()
        seg_detail = st.empty()
        
        if col3_diplay:
            # st.markdown('<h3 style="color:#ADD8E6;">Analyzing Results...</h3>', unsafe_allow_html=True)
            # Create a container for the columns
            seg_detail.markdown('<h3>Segmentation Details</h3>', unsafe_allow_html=True)

            with st.container():
                col1,col2,col3, col4, col5=st.columns(5,gap='small')
                with col1:

                    st.metric(label="Affected Area",value=f"{int(total_affected_area)} cm²")

                with col2:
                    st.metric(label="Entropy",value=f"{int(total_entropy)}")
                with col3:
                    st.metric(label="Total Perimeter",value=f"{int(total_perimeter)}")

                with col4:
                    st.metric(label="Mean",value=f"{int(mean_area)} cm²")
                with col5:
                    st.metric(label="RMS",value=f"{int(rms_area)} cm²")
            # st.markdown('<h3 style="color:#ADD8E6;">Analyzing Results...</h3>', unsafe_allow_html=True)
            st.divider()
            
            placeholder = st.empty()
            placeholder.markdown('<h3 style="color:#ADD8E6;">Analyzing Results...</h3>', unsafe_allow_html=True)
            # placeholder.title("Analyzing Results...")
            precaution=precaution_bot(class_name, num_detections, detection_details)
            if precaution:
                # st.title("Disease Precautions!")
                placeholder.markdown('<h3>Disease Precautions!</h3>', unsafe_allow_html=True)

                with st.status("", expanded=True) as status:
                    if on:
                        text=precaution_convert_into_arabic("https://deep-translator-api.azurewebsites.net/google/",{"source": "english","target": "arabic","text": f"{precaution}","proxies": []})
                        st.write(text)
                    else:
                        st.write(precaution)
            else:
                placeholder.markdown('<h3>No Disease Detected!</h3>', unsafe_allow_html=True)


elif options=="Live Camera":
    st.markdown('<h3>Take a Image</h3>', unsafe_allow_html=True)
    image=st.camera_input("Take a Picture")
    
    if image is not None:
        st.image(image)
         # Read the uploaded file as an OpenCV image
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_width_pixels = 300
        image_height_pixels = 300
        # Define the pixel density (DPI)
        dpi = 96
         # Define the desired size
        desired_size = (image_width_pixels, image_height_pixels)  # Width, Height


        # Convert pixels to centimeters
        image_width_cm = (image_width_pixels / dpi) * 2.54
        image_height_cm = (image_height_pixels / dpi) * 2.54
        # Resize both images
        image_rgb_resized = cv2.resize(image_rgb, desired_size)
        gray_image_resized = cv2.resize(gray_image, desired_size)
        # Binarize the grayscale image
        _, binary_image_resized = cv2.threshold(gray_image_resized, 128, 255, cv2.THRESH_BINARY)

        results = list(model(image_rgb_resized))  # Pass the RGB image to the model
        # Access the first result (if there's only one image processed)
        result = results[0]
        # Get the boxes attribute which contains the confidence scores
        boxes = result.boxes
        class_name = ''
        num_detections = 0
        detection_details = ''
        segmented_areas = []
        total_affected_area = 0
        total_entropy = 0
        mean_area = 0
        rms_area = 0
        total_perimeter = 0

        # Check if there are any boxes
        if boxes and boxes.conf.numel() > 0:
            # Print the number of detections
            num_detections = boxes.conf.numel()
            print(f"Number of detections: {num_detections}")

            # Find the index of the box with the highest confidence
            max_confidence_index = boxes.conf.argmax()

            # Get the class ID of the box with the highest confidence
            class_id = boxes.cls[max_confidence_index].item()

            # Get the class name using the class ID
            class_name = result.names[class_id]

            print(f"Class with highest confidence: {class_name}")

            pixel_area_cm = (image_width_cm / image_width_pixels) * (image_height_cm / image_height_pixels)
            total_image_area_cm = image_width_cm * image_height_cm


            total_entropy = 0

            for i, confidence in enumerate(boxes.conf):
                mask = result.masks.data[i]
                area_in_pixels = mask.sum().item()
                area_in_cm = area_in_pixels * pixel_area_cm
                segmented_areas.append(area_in_cm)
                detection_details += f"Detection {i + 1}: with Confidence = {confidence.item()} and Segmented Area = {area_in_cm} cm² "
                # Calculate entropy for each mask and add to total entropy
                entropy_value = shannon_entropy(mask.cpu().numpy())
                total_entropy += entropy_value
                # Calculate perimeter
                label = measure.label(mask.cpu().numpy())
                props = measure.regionprops(label)
                for prop in props:
                    total_perimeter +=prop.perimeter
                    print(f"Detection {i + 1}: Perimeter = {total_perimeter:.3f}")
            print(detection_details)

            # Calculate total affected area
            total_affected_area = sum(segmented_areas)
            # Calculate mean and RMS of segmented areas
            mean_area = np.mean(segmented_areas)
            rms_area = np.sqrt(np.mean(np.square(segmented_areas)))

            # Round the values to 3 decimal places
            total_affected_area = round(total_affected_area, 3)
            total_entropy = round(total_entropy, 3)
            mean_area = round(mean_area, 3)
            rms_area = round(rms_area, 3)

            print(f"Total Affected Area: {total_affected_area} cm²")
            print(f"Total Entropy Value: {total_entropy}")
            print(f"Total Total Perimeter: {total_perimeter} ")
            print(f"Mean Segmented Area: {mean_area} cm²")
            print(f"RMS Segmented Area: {rms_area} cm²")

        else:
            print("No objects detected.")

        # print('-------- results', results)
        # result = results[0]
        # no_of_masks, x, y = result.masks.shape
        st.divider()
        st.markdown('<h3>Image Analysis</h3>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
       
        # Add the first row of columns for the RGB and grayscale images
        # col1, col2 = st.columns(2)
        col3_diplay = False


        # Display the RGB image in the first column
        with col1:
            st.image(image_rgb_resized, channels="RGB", caption='RGB Image')

        # Display the grayscale image in the second column
        with col2:
            st.image(gray_image_resized, caption='Grayscale Image')

        # Add a second row of columns for the segmented image and binary mask
        # col3, col4 = st.columns(2)

        # Display the segmented image in the third column
        with col3:
            res_plotted = result.plot(line_width=1, labels=True)  # Ensure result is in RGB
            class_caption = f"'{class_name}' disease detected"
            st.image(res_plotted, caption=class_caption, use_column_width=True)

        # Create and display the binary mask in the fourth column
        with col4:
            if boxes and boxes.conf.numel() > 0:
                # Create an empty binary mask
                binary_mask = np.zeros_like(image_rgb_resized[:, :, 0], dtype=np.uint8)

                for i in range(len(boxes.conf)):
                    mask = result.masks.data[i].cpu().numpy()
                    # Resize the mask to match the binary mask dimensions
                    resized_mask = cv2.resize(mask, (binary_mask.shape[1], binary_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    binary_mask[resized_mask > 0] = 255  # Set mask areas to white
                col3_diplay = True
                st.image(binary_mask, caption='Affected Area', use_column_width=True)
                
            else:
                st.write("No objects detected.")
        st.divider()
        seg_detail = st.empty()
        
        if col3_diplay:
            # st.markdown('<h3 style="color:#ADD8E6;">Analyzing Results...</h3>', unsafe_allow_html=True)
            # Create a container for the columns
            seg_detail.markdown('<h3>Segmentation Details</h3>', unsafe_allow_html=True)

            with st.container():
                col1,col2,col3, col4, col5=st.columns(5,gap='small')
                with col1:

                    st.metric(label="Affected Area",value=f"{int(total_affected_area)} cm²")

                with col2:
                    st.metric(label="Entropy",value=f"{int(total_entropy)}")
                with col3:
                    st.metric(label="Total Perimeter",value=f"{int(total_perimeter)}")

                with col4:
                    st.metric(label="Mean",value=f"{int(mean_area)} cm²")
                with col5:
                    st.metric(label="RMS",value=f"{int(rms_area)} cm²")
            # st.markdown('<h3 style="color:#ADD8E6;">Analyzing Results...</h3>', unsafe_allow_html=True)
            st.divider()
            
            placeholder = st.empty()
            placeholder.markdown('<h3 style="color:#ADD8E6;">Analyzing Results...</h3>', unsafe_allow_html=True)
            # placeholder.title("Analyzing Results...")
            precaution=precaution_bot(class_name, num_detections, detection_details)
            if precaution:
                # st.title("Disease Precautions!")
                placeholder.markdown('<h3>Disease Precautions!</h3>', unsafe_allow_html=True)

                with st.status("", expanded=True) as status:
                    if on:
                        text=precaution_convert_into_arabic("https://deep-translator-api.azurewebsites.net/google/",{"source": "english","target": "arabic","text": f"{precaution}","proxies": []})
                        st.write(text)
                    else:
                        st.write(precaution)
            else:
                placeholder.markdown('<h3>No Disease Detected!</h3>', unsafe_allow_html=True)


elif options=="AI-Agent":
    
    # Initialize the chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    # Sidebar layout with container for messages
    with st.sidebar:
        # Define the container for chat messages
        messages_container = st.container(height=500)
        
        # Chat input at the bottom
        prompt = st.chat_input("Say something")
        if on:
            if prompt:
                response = precaution_chatbot(prompt)  # Get response from chatbot
                convert_arabic_prompt=precaution_convert_into_arabic("https://deep-translator-api.azurewebsites.net/google/",{"source": "english","target": "arabic","text": f"{prompt}","proxies": []})
                convert_arabic_response=precaution_convert_into_arabic("https://deep-translator-api.azurewebsites.net/google/",{"source": "english","target": "arabic","text": f"{response}","proxies": []})
                # Append user and bot messages to session state
                st.session_state.messages.append({"role": "user", "content": convert_arabic_prompt})
                st.session_state.messages.append({"role": "assistant", "content": convert_arabic_response})
        else:
            if prompt:
                response = precaution_chatbot(prompt)  # Get response from chatbot
                # Append user and bot messages to session state
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response})

        # Display chat history inside the container
        with messages_container:
            if st.session_state.messages:
                for message in st.session_state.messages:
                    if message['role'] == 'user':
                        st.chat_message("user").write(message['content'])
                    else:
                        st.chat_message("assistant").write(message['content'])
