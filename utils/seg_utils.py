import io
import os
import typing

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from configs.seg_config import LAYERS_CONFIG


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
                new_value = st.selectbox(
                    key, options=options, index=options.index(default_value)
                )  # Fixed the argument duplication
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


def select_layer_seg_aug():
    st.subheader("Select a Layer")
    layer_option = st.selectbox(
        "Select an option", list(LAYERS_CONFIG.keys()), index=0, key="layer_option"
    )
    layer_config = LAYERS_CONFIG[layer_option]

    # Extract relevant information from layer_config
    layer_cls = layer_config["layer_cls"]
    layer_args = layer_config["layer_args"]
    control_args = layer_config["control_args"]

    # Set control arguments
    layer_args = set_control_args(control_args, layer_args)

    # Instantiate layer
    layer = layer_cls(**layer_args)

    return layer


def download_images_seg():
    root_dir = "images/seg/"
    default_images = {}

    image_data = {
        "city": {"image_name": "image_1.png", "mask_name": "mask_1.png"},
    }

    for name, image_info in image_data.items():
        image_path = os.path.join(root_dir, image_info["image_name"])
        mask_path = os.path.join(root_dir, image_info["mask_name"])

        with open(image_path, "rb") as f:
            image_content = f.read()
        with open(mask_path, "rb") as f:
            mask_content = f.read()

        default_images[name] = {
            "image": image_content,
            "mask": mask_content,
        }

    return default_images


def image_dropdown(image_dict=None):
    if image_dict is None:
        image_dict = download_images_seg()

    st.subheader("Upload Image and Mask")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    uploaded_mask = st.file_uploader("Upload Mask", type=["jpg", "jpeg", "png"])

    if uploaded_image and uploaded_mask:
        image = Image.open(uploaded_image).convert("RGB")
        mask = Image.open(uploaded_mask).convert("RGB")
        return np.array(image), np.array(mask)

    st.subheader("Or select an existing image")
    image_option = st.selectbox(
        "Select an option",
        list(image_dict.keys()),
        index=0,
        key="image_option",
    )

    image_data = image_dict[image_option]
    image = Image.open(io.BytesIO(image_data["image"])).convert("RGB")
    mask = Image.open(io.BytesIO(image_data["mask"])).convert("RGB")
    return np.array(image), np.array(mask)


def Preprocessing(layer, image, mask):
    """
    Input Format : {"images": tf.cast(image, tf.float32),
                    "segmentation_masks": mask}
    """
    image = image.copy()
    mask = mask.copy()

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)

    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=0)

    inputs = {"images": image, "segmentation_masks": mask}

    outputs = layer(inputs)
    return inputs, outputs


def display_img_with_mask(inputs, outputs):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image and Mask")
        st.image(
            np.array(inputs["images"]).astype(np.uint8),
            use_column_width=True,
            caption="Image",
        )
        st.image(
            np.array(inputs["segmentation_masks"]).astype(np.uint8),
            use_column_width=True,
            caption="Mask",
        )

    with col2:
        st.subheader("Output Image and Mask")
        st.image(
            np.array(outputs["images"]).astype(np.uint8),
            use_column_width=True,
            caption="Augmented Image",
        )
        st.image(
            np.array(outputs["segmentation_masks"]).astype(np.uint8),
            use_column_width=True,
            caption="Augmented Mask",
        )


def seg():
    image, mask = image_dropdown()
    layer = select_layer_seg_aug()
    inputs, outputs = Preprocessing(layer, image, mask)
    return inputs, outputs
