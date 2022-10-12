import streamlit as st

#-----HEADER------

with st.container():
    st.title("IBODS Project Deployement")
    st.write("Esta es la pagina de pruebas para nuestro proyecto")

thr = st.sidebar.slider("Detection Threshold", min_value = 0.0, max_value = 1.0, value = 0.3, step = 0.01)

model = st.sidebar.selectbox("Select Model",  ("EfficientDet0", "EfficientDet1"))