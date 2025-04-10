import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import streamlit as st
from PIL import Image, ImageColor

from makeup import sharpen,hair


st.set_page_config(
    page_title="ColorPop",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEMO_IMAGE = 'imgs/deepika.jpg'

st.title('ColorPop : Find Your Perfect Hue.')
st.write('ColorPop is an innovative feature within **Flipkart Grid 4.0** that uses **BisNet technology** to map lipstick and hair colors to **facial feature segmentation**, allowing customers to virtually try on products using their own photos. It solves the problem of uncertainty in color selection by providing a realistic, personalized experience that helps users see how shades will look on them before purchasing. By offering this virtual try-on, ColorPop reduces returns, saves time, and enhances confidence in beauty product choices, all from the comfort of home.')
st.markdown("<h3 style='text-align: center; color: black;'>Try It Now!</h3>", unsafe_allow_html=True)
st.divider()

st.sidebar.title("🎨 ColorPop : Find Your Perfect Hue.")
st.sidebar.header("🔧 Adjust Your Preferences")

table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13,
        
    }

img_file_buffer = st.sidebar.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    demo_image = img_file_buffer

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))
    
new_image = image.copy()
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Original Image")
    st.image(image,use_container_width = True)

cp = 'checkpoint/79999_iter.pth'
ori = image.copy()
h,w,_ = ori.shape

image = cv2.resize(image,(1024,1024))
parsing = evaluate(demo_image, cp)
parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

parts = [table['hair'], table['upper_lip'], table['lower_lip']]

hair_color = st.sidebar.color_picker('Pick Hair Color', '#000')
hair_color = ImageColor.getcolor(hair_color, "RGB")

lip_color = st.sidebar.color_picker('Pick Lip Color', '#580029')
lip_color = ImageColor.getcolor(lip_color, "RGB")



colors = [hair_color, lip_color, lip_color]

for part, color in zip(parts, colors):
    image = hair(image, parsing, part, color)

image = cv2.resize(image,(w,h))

with col2:
    st.subheader("✨ Transformed Image")
    st.image(image,use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 20px;'>Made with love by Team Zed: Kishan Mishra, Satish Sahu</p>", unsafe_allow_html=True)