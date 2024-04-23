import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time

st.set_page_config(layout="wide")

cfg_model_path = 'models/yolov5s.pt'
model = None
confidence = .25


def image_input(data_src):
    img_file = None
    if data_src == '样例数据':
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("选择一张测试图片", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    elif data_src == '电动车头盔检测数据':
        img_path = glob.glob('data/sample_helmet_detection_images/*')
        img_slider = st.slider("选择一张测试图片", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("上传一张图片", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="已选择的图片")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="模型预测结果")


def video_input(data_src):
    vid_file = None
    if data_src == '样例数据':
        vid_file = "data/sample_videos/sample.mp4"
    elif data_src == '电动车头盔检测数据':
        vid_file = "data/sample_videos/sample_helmet.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("上传一个视频", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("自定义框架尺寸")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                # st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()


def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
    # , force_reload=True
    model_.to(device)
    print("已切换到:", device)
    return model_


@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def get_user_model():
    model_src = st.sidebar.radio("模型来源", ["文件上传", "url"])
    model_file = None
    if model_src == "文件上传":
        model_bytes = st.sidebar.file_uploader("上传一个模型文件", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("模型地址")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_

    return model_file


def camera_input():
    st.header("摄像头实时监测")
    cap = cv2.VideoCapture(0)  # 默认摄像头
    frame_window = st.image([])

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("从摄像头中获取帧失败，请检查摄像头是否正常")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_img = infer_image(frame)  # 使用模型处理帧
        frame_window.image(output_img)  # 显示处理后的帧


def main():
    # global variables
    global model, confidence, cfg_model_path

    st.title("智安骑行——基于计算机视觉的城市电动车骑行头盔智能检测与识别系统")

    st.sidebar.title("设置")

    # upload model
    model_src = st.sidebar.radio("选择YOLOv5权重文件", ["使用官方的5s模型", "电动车头盔检测模型", "使用自己的模型"])
    # URL, upload file (max 200 mb)
    if model_src == "使用自己的模型":
        user_model_path = get_user_model()
        if user_model_path:
            cfg_model_path = user_model_path

        st.sidebar.text(cfg_model_path.split("/")[-1])
        st.sidebar.markdown("---")
    elif model_src == "电动车头盔检测模型":
        cfg_model_path = "models/uploaded_v5s_helmet_voc2007_best.pt"

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("选择设备", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("选择设备", ['cpu', 'cuda'], disabled=True, index=0)

        # load model
        model = load_model(cfg_model_path, device_option)

        # confidence slider
        confidence = st.sidebar.slider('置信度', min_value=0.1, max_value=1.0, value=.45)

        # custom classes
        if st.sidebar.checkbox("自定义种类"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("选择种类", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        # input options
        input_option = st.sidebar.radio("选择输入类型: ", ['图片', '视频', '摄像头'])

        # input src option
        data_src = st.sidebar.radio("选择输入来源: ", ['样例数据', '电动车头盔检测数据','上传数据'])

        if input_option == '图片':
            image_input(data_src)
        elif input_option == '视频':
            video_input(data_src)
        elif input_option == '摄像头':
            camera_input()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
