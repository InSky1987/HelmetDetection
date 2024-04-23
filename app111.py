
import streamlit as st
import torch
from PIL import Image
import cv2
import os
import glob
import wget
import numpy as np

st.set_page_config(layout="wide")

cfg_model_path = 'models/yolov5s.pt'
model = None
confidence = .25


def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
    model_.to(device)
    st.write("Model loaded on:", device)
    return model_


def infer_image(img, size=None):
    if isinstance(img, str):  # If image path is given
        img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Perform inference using the model
    results = model(img)
    # Render the detections on the image and retrieve the modified images
    results.render()  # Draw the boxes on image
    img_array = results.ims[0]  # imgs is now correctly accessed after calling render
    # Convert the array back to an image
    img = Image.fromarray(img_array)
    return img




def camera_input():
    st.header("Camera Live Feed")
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])

    # Use a static key for the button since this is the only instance in this context
    stop_button = st.sidebar.button("Stop Camera", key="stop_camera")

    while True:
        ret, frame = cap.read()
        if not ret or stop_button:
            break
        output_img = infer_image(frame)
        frame_window.image(output_img)
    cap.release()
    st.write("Camera stopped")



def image_input(data_src):
    img_file = None
    if data_src == '样例数据':
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Choose a sample image", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Model Prediction")


def video_input(data_src):
    vid_file = None
    if data_src == '样例数据':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        frame_window = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or st.sidebar.button("Stop Video"):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            frame_window.image(output_img)
        cap.release()
        st.write("Video stopped")


def main():
    global model, confidence

    st.title("Real-time Object Detection with YOLOv5")

    # Model and device setup
    device_option = st.sidebar.radio("Select device", ['cpu', 'cuda', 'camera'], index=0)

    if not os.path.isfile(cfg_model_path):
        st.error("Model file not available!!! Please add it to the model folder.")
        return

    if device_option in ['cpu', 'cuda']:  # Load model for CPU or CUDA
        model = load_model(cfg_model_path, device_option)
        confidence = st.sidebar.slider('Confidence threshold', min_value=0.1, max_value=1.0, value=.25)

    # Setup for camera needs the model too
    if device_option == 'camera':
        if model is None:
            model = load_model(cfg_model_path, 'cpu')  # Default to CPU if camera is selected without a model loaded
        camera_input()
    else:
        data_src = st.sidebar.radio("Select data source:", ['样例数据', '上传数据'])
        input_option = st.sidebar.radio("Select input type:", ['图片', '视频'])

        if input_option == '图片':
            image_input(data_src)
        elif input_option == '视频':
            video_input(data_src)

if __name__ == "__main__":
    main()




