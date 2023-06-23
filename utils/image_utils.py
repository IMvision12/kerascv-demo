import io
import os
import typing

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from configs.img_config import LAYERS_CONFIG


def download_images():
    image_folder = "images/"
    default_images = {}

    for filename in os.listdir(image_folder):
        if (
            filename.endswith(".jpg")
            or filename.endswith(".jpeg")
            or filename.endswith(".png")
        ):
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
    layer_option = st.selectbox(
        "Select an option", list(LAYERS_CONFIG.keys()), index=0, key="layer_option"
    )
    layer_cls = LAYERS_CONFIG[layer_option]["layer_cls"]
    layer_args = LAYERS_CONFIG[layer_option]["layer_args"]
    control_args = LAYERS_CONFIG[layer_option]["control_args"]
    layer_args = set_control_args(control_args, layer_args)
    layer = layer_cls(**layer_args)
    return layer


def set_control_args(control_args: typing.Dict, layer_args: typing.Dict):
    """Use `st.select_slider` or `st.slider` for `control_args` depending on
    default value.
    """
    with st.form(key="control"):
        new_values = {}
        for key, value in control_args.items():
            if isinstance(layer_args[key], str):
                options = value
                default_value = layer_args[key]
                new_value = st.selectbox(key, options=options, index=options.index(default_value))  # Fixed the argument duplication
            else:
                min_value = value[0]
                max_value = value[1]
                default_value = layer_args[key]
                new_value = st.slider(key, min_value, max_value, default_value)
            new_values[key] = new_value
        submit_button = st.form_submit_button(label="Apply")
        if submit_button:
            layer_args.update(new_values)
    return layer_args