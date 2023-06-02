import io
import os
import typing

import tensorflow as tf
import numpy as np
import streamlit as st
from config import LAYERS_CONFIG
from PIL import Image


@st.cache_data

def download_images():
    image_folder = "F:\keras\kerascv-demo\images"
    default_images = {}
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            with open(image_path, "rb") as f:
                default_images[filename] = f.read()
    
    return default_images



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


def process_image(image, layer):
    processed_image = layer(image)
    processed_image: np.ndarray = processed_image.numpy()
    processed_image = np.round(processed_image).astype(np.uint8)
    return processed_image


def main():
    st.set_page_config(
        page_title="KerasCV Demo Site",
        initial_sidebar_state="expanded",
        layout="wide",
    )

    st.title("KerasCV Demo Site")
    st.write("Welcome to the KerasCV Demo Site!")
    st.write('Press "R" to generate a new random image')

    data_load_state = st.text("Loading data...")
    default_images = download_images()
    data_load_state.empty()

    with st.sidebar:
        # Images
        st.subheader("Select an Image")
        image_option = st.selectbox(
            "",
            ["Default Image"] + list(default_images.keys()),
            index=0,
            key="image_option",
        )

        uploaded_image = None
        if image_option == "Default Image":
            image = Image.open(io.BytesIO(default_images["cat.jpeg"]))
        else:
            with st.expander("Upload an image"):
                uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
            else:
                image = Image.open(io.BytesIO(default_images[image_option]))

        # Layers
        st.subheader("Select a Layer")
        layer_option = st.selectbox("", list(LAYERS_CONFIG.keys()), index=0, key="layer_option")
        layer_cls = LAYERS_CONFIG[layer_option]["layer_cls"]
        layer_args = LAYERS_CONFIG[layer_option]["layer_args"]
        control_args = LAYERS_CONFIG[layer_option]["control_args"]
        layer_args = set_control_args(control_args, layer_args)
        layer = layer_cls(**layer_args)

    col1, col2, col3 = st.columns([1, 0.1, 1])

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col3:
        st.subheader("Processed Image")
        processed_image = process_image(image, layer)
        st.image(processed_image, use_column_width=True)

    # Show help
    with st.expander(f"Click to display help for {layer_cls.__name__}"):
        st.help(layer)


if __name__ == "__main__":
    try:
        tf.config.set_visible_devices([], "GPU")
    except:
        pass

    main()