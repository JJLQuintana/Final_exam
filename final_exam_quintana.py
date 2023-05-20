import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('agricultural_crop_classifier.h5')
  return model
model=load_model()
st.write("""
# Agricultural Crop Classification"""
)
file=st.file_uploader("Choose Agricultural crop photo from computer.",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(130,130)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image related to weather.")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names = ['coffee-plant', 'lemon', 'banana']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
