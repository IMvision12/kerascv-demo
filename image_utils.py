import io
import os
import typing

import tensorflow as tf
import numpy as np
import streamlit as st
from config import LAYERS_CONFIG
from PIL import Image
from utils import set_control_args


def download_images():
    image_folder = "images/"
    default_images = {}
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            with open(image_path, "rb") as f:
                default_images[filename] = f.read()
    return default_images


def process_image(image, layer):
    processed_image = layer(image)
    processed_image: np.ndarray = processed_image.numpy()
    processed_image = np.round(processed_image).astype(np.uint8)
    return processed_image


def image_aug(image_dict=download_images()):
    st.subheader("Select an Image")
    image_option = st.selectbox(
            "Select an option",
            ["Default Image"] + list(image_dict.keys()),
            index=0,
            key="image_option",
    )
    uploaded_image = None
    if image_option == "Default Image":
        image = Image.open(io.BytesIO(image_dict["cat.jpeg"]))
    else:
        with st.expander("Upload an image"):
            uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
        else:
            image = Image.open(io.BytesIO(image_dict[image_option]))

    layer = select_layer_for_image_aug()
    return layer, image


def display_aug_image(layer, image):
    col1, col2, col3 = st.columns([1, 0.1, 1])

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col3:
        st.subheader("Processed Image")
        processed_image = process_image(image, layer)
        st.image(processed_image, use_column_width=True)


def select_layer_for_image_aug():
    st.subheader("Select a Layer")
    layer_option = st.selectbox("Select an option", list(LAYERS_CONFIG.keys()), index=0, key="layer_option")
    layer_cls = LAYERS_CONFIG[layer_option]["layer_cls"]
    layer_args = LAYERS_CONFIG[layer_option]["layer_args"]
    control_args = LAYERS_CONFIG[layer_option]["control_args"]
    layer_args = set_control_args(control_args, layer_args)
    layer = layer_cls(**layer_args)
    return layer