import cv2
import numpy as np
import streamlit as st


st.write("""
    # Let me sketch your photo! 

    Please upload any photo you have.

""")

uploaded_file = st.file_uploader(
    "Upload your file", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    #st.image(opencv_image, channels="BGR")


def dodge(x, y):
    return cv2.divide(x, 255 - y, scale=256)


def burn_img(image, mask):
    return 255 - cv2.divide(255-image, 255-mask, scale=256)


def sketch(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_invert = cv2.bitwise_not(img_gray)

    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)

    dodged_img = dodge(img_gray, img_smoothing)

    final_img = burn_img(dodged_img, img_smoothing)

    return final_img


def Colorsketch(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_invert = cv2.bitwise_not(img)

    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)

    dodged_img = dodge(img, img_smoothing)

    final_img = burn_img(dodged_img, img_smoothing)

    return final_img


st.title('Sketches with Python')

col1, col2, col3 = st.beta_columns(3)


col1.header("Original")
col1.image(opencv_image, use_column_width=True, channels="BGR")

col2.header("Sketch")
col2.image(sketch(opencv_image), use_column_width=True)

col3.header("Color Sketch")
col3.image(Colorsketch(opencv_image), use_column_width=True)
