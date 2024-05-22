import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import cv2

genai.configure(api_key="")  # Replace with your Gemini API key

## Function to load Google Gemini Pro Vision API
def get_gemini_response(input_prompt, image_data, input_text):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    response = model.generate_content([input_prompt, image_data, input_text])
    return response.text

## Function to process uploaded image
def process_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

## Function to capture image from camera (single shot)
def capture_camera_image():
    cap = cv2.VideoCapture(1)  # Open rear camera (index 1)

    # Check if camera opened successfully
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    # Capture a single frame
    ret, frame = cap.read()

    # Display the captured frame (optional)
    # cv2.imshow('Camera', frame)
    # cv2.waitKey(0)

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Convert frame to RGB for compatibility
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image for processing
    image = Image.fromarray(image)

    # Convert PIL Image to list format for Gemini API
    image_parts = [{'mime_type': 'image/jpeg', 'data': image.tobytes()}]
    return image_parts

## Initialize Streamlit app
st.set_page_config(page_title="Gemini Health App")

st.header("Gemini Health App")

# Option to choose between upload or camera
image_source = st.selectbox("Select Image Source", ("Upload Image", "Capture from Camera"))

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
else:
    try:
        # Capture a single image on button click
        if st.button("Capture Image"):
            image_parts = capture_camera_image()
    except RuntimeError as e:
        st.error(f"Error capturing camera image: {e}")
        image_parts = None

input_text = st.text_input("Input Prompt:", key="input")

submit = st.button("Tell me the total calories")

input_prompt = """
You are an expert in nutritionist where you need to see the food items from the image
and calculate the total calories, also provide the details of every food items with calories intake
is below format

1. Item 1 - no of calories
2. Item 2 - no of calories
----
----


"""

if submit:
    if image_parts is None:
        st.error("Please upload an image or capture one from the camera.")
    else:
        response = get_gemini_response(input_prompt, image_parts, input_text)
        st.subheader("The Response is")
        st.write(response)
