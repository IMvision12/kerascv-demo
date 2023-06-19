import io
import os
import typing

import tensorflow as tf
import numpy as np
import streamlit as st
from config import LAYERS_CONFIG
from PIL import Image
from image_utils import image_aug, display_aug_image
from bbox_utils import bbox, display_img_with_bbox

def main():
    st.set_page_config(
        page_title="KerasCV Demo Site",
        initial_sidebar_state="expanded",
        layout="wide",
    )

    st.title("KerasCV Demo Site")
    st.write("Welcome to the KerasCV Demo Site!")
    st.write('Press "R" to generate a new random image')

    with st.sidebar:
        st.subheader("Choose Agumentation Type: ")
        option = st.selectbox("Select an option", ("Image", "Bounding-Box", "Segmentation"))

        if option == "Image":
            layer, image = image_aug()
        if option == "Bounding-Box":
            layer, image, box, box_format = bbox()

    if option == "Image":
        display_aug_image(layer, image)
    if option == "Bounding-Box":
        display_img_with_bbox(image, box, layer, box_format)


if __name__ == "__main__":
    try:
        tf.config.set_visible_devices([], "GPU")
    except:
        pass

    main()