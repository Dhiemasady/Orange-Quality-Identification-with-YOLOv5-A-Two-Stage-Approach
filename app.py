import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import detect
from detect import run
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
st.set_page_config(layout="wide")

cfg_model_path = 'models/tester.pt'
model = None
model_jeruk = None
confidence = .6
model_jeruk_path = 'models/jeruk_besar.pt'

def image_input(data_src):
    img_file = None
    model_jeruk = torch.hub.load('.', 'custom', path=model_jeruk_path, source='local', force_reload=True) #model untuk deteksi
    model = torch.hub.load('.', 'custom', path=cfg_model_path, source='local', force_reload=True) #model untuk konvolusi

    img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    if img_bytes:
        img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
        Image.open(img_bytes).save(img_file)

    if img_file:
        jeruk = model_jeruk(img_file)
        jeruk = jeruk.pandas().xyxy[0] #jumlah terdeteksi objek pada citra
        jml_jrk = 0 

        for i in jeruk.index: #jumlah jeruk
            if jeruk['class'][i] == 1:
                jml_jrk+=1

        if jml_jrk > 0: 
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_file, caption="Selected Image") #menampilkan gambar asli
            with col2:
                img = model(img_file) #model deteksi
                img.show() #menampilkan deteksi
            run(source=img_file, weights=cfg_model_path) #Menampilkan hasil konvolusi dari proses deteksi
        else:
            st.markdown("<h3>Terdeteksi objek namun bukan jeruk penelitian </h3>", unsafe_allow_html=True)

def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image

def main():
    # global variables
    global model, confidence, cfg_model_path

    st.title("Pengujian Model Deteksi Dan Klasifikasi Kualitas Buah Jeruk")

    #st.sidebar.title("Settings")

    # upload model
    #model_src = st.sidebar.radio("Select yolov5 weight file", ["Use your own model"])
    # URL, upload file (max 200 mb)
    #if model_src == "Use your own model":
    #    user_model_path = get_user_model()
    #    if user_model_path:
    #        cfg_model_path = user_model_path

    #    st.sidebar.text(cfg_model_path.split("/")[-1])
    #st.sidebar.markdown("---")

    # check if model file is available
    device_option = 'cpu'
        # load model
  

        # confidence slider
    


    #st.sidebar.markdown("---")

        # input options
    input_option = 'image'

        # input src option
    data_src = st.sidebar.radio("Select input source: ", ['Upload Data Pengujian'])

    if input_option == 'image':
        image_input(data_src)

#main menu
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
