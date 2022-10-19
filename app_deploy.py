from cProfile import run
import streamlit as st
from PIL import Image
from helper_functions import run_odt_and_draw_results

#-----HEADER------

with st.container():
    st.title("IBODS Project Deployement")
    st.write("Esta es la pagina de pruebas para nuestro proyecto")

thr = st.sidebar.slider("Detection Threshold", min_value = 0.0, max_value = 1.0, value = 0.3, step = 0.01)

model = st.sidebar.selectbox("Select Model",  ("EfficientDet0", "EfficientDet1"))

image_file = st.file_uploader("Upload images for object detection", type=['png','jpeg'])

if image_file is not None:
    input_image = Image.open(image_file)
    st.image(input_image)

detect = st.button("Detect objects")

if detect:
    run_odt_and_draw_results(input_image)
