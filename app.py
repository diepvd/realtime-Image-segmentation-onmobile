import streamlit as st
import numpy as np
# import tensorflow as tf
import os
from PIL import Image,ImageColor
import cv2
import altair as alt
import tensorflow.keras as k
from utils import apply_color, resize_mask

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

model_path='weights/hair_segmentation_mobile.h5'
INPUT_SHAPE=(192,192)


st.title("Hair Color")

@st.cache
def load_model():
    model = k.models.load_model(model_path, compile=False)
    return model

#laoding model
model = load_model()  
st.success("Model loaded")

#Code for uploading file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image=image/255.

#Adding color picker
color = st.beta_color_picker('Pick A Color', '#00f900')
st.write('The current color is', color)

#Covert hex to rgb
color = ImageColor.getrgb(color)

#Prediction
input_image = cv2.resize(image,INPUT_SHAPE)
input_image = np.expand_dims(input_image, axis=0)
prediction = model.predict(input_image)[..., 0:1]

#Applying color
prediction = apply_color(image, prediction[0], list(color))
prediction = (prediction*255.).astype(np.uint8)

#Displaying output
st.image(prediction,use_column_width=True)