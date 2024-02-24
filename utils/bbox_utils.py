import io
import os
import cv2
import typing

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from keras_cv.visualization import draw_bounding_boxes
from PIL import Image

from configs.bbox_config import LAYERS_CONFIG


def download_images_bbox():
    root_dir = "images/"
    default_images = {}

    image_data = {
        "cat": {
            "filename": "cat.jpeg",
            "boxes": [
                {"x": 6, "y": 116, "w": 844, "h": 1384},
            ],
        },
        "dog": {
            "filename": "dog.jpg",
            "boxes": [
                {"x": 349, "y": 283, "w": 633, "h": 699},
            ],
        },
        "fish": {
            "filename": "fish.jpg",
            "boxes": [
                {"x": 229, "y": 136, "w": 153, "h": 274},
                {"x": 99, "y": 213, "w": 157, "h": 186},
                {"x": 394, "y": 208, "w": 256, "h": 208},
                {"x": 613, "y": 171, "w": 235, "h": 196},
                {"x": 787, "y": 383, "w": 121, "h": 179},
                {"x": 368, "y": 399, "w": 197, "h": 193},
                {"x": 554, "y": 351, "w": 213, "h": 249},
            ],
        },
    }

    for image_name, image_info in image_data.items():
        image_path = os.path.join(root_dir, image_info["filename"])
        with open(image_path, "rb") as f:
            image_content = f.read()

        boxes_df = pd.DataFrame(image_info["boxes"])

        default_images[image_name] = {
            "image": image_content,
            "boxes": boxes_df,
        }

    return default_images


def image_dropdown(image_dict=download_images_bbox()):
    st.subheader("Select an Image")
    image_option = st.selectbox(
        "Select an option",
        list(image_dict.keys()),
        index=0,
        key="image_option",
    )

    with st.expander("Upload an image"):
        uploaded_image = st.file_uploader(
            "dummy",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

    if uploaded_image is not None:
        st.subheader("Select BBox Type")
        options = st.selectbox(
            "Select an option",
            ["xywh", "xyxy"],
            index=0,
        )
        image = Image.open(uploaded_image).convert("RGB")

        if options == "xywh":
            boxes = pd.DataFrame({"x": [0], "y": [0], "w": [0], "h": [0]})
        elif options == "xyxy":
            boxes = pd.DataFrame({"x": [0], "y": [0], " x": [0], " y": [0]})
    else:
        image_data = image_dict[image_option]
        image = Image.open(io.BytesIO(image_data["image"])).convert("RGB")
        boxes = image_data["boxes"]

    return np.array(image), boxes


def Preprocessing(layer, image, box_format="xywh", boxes=None):
    """
    bounding_boxes = {
        # num_boxes may be a Ragged dimension
        'boxes': Tensor(shape=[batch, num_boxes, 4]),
        'classes': Tensor(shape=[batch, num_boxes])
    }

    Input Format : {"images": tf.cast(image, tf.float32),
                    "bounding_boxes": bounding_boxes}

    Reference : https://keras.io/guides/keras_cv/object_detection_keras_cv/
    """
    image = image.copy()
    inputs = {
        "images": tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0)
    }

    if boxes is not None:
        boxes = tf.expand_dims(tf.convert_to_tensor(boxes, dtype=tf.float32), axis=0)
        inputs["bounding_boxes"] = {
            "boxes": boxes,
            "classes": tf.zeros(shape=boxes.shape[:-1]),
        }

    bounding_boxes = inputs.get("bounding_boxes", {}).copy()
    input_image = draw_bounding_boxes(
        inputs["images"],
        bounding_boxes,
        color=(0, 224, 0),
        bounding_box_format=box_format,
    )

    outputs = layer(inputs)
    if "bounding_boxes" in outputs:
        output_image = draw_bounding_boxes(
            outputs["images"],
            outputs["bounding_boxes"],
            color=(0, 224, 0),
            bounding_box_format=box_format,
        )
    else:
        output_image = np.array(outputs["images"]).astype(np.uint8)
    return input_image, output_image


def select_layer_bbox_aug():
    st.subheader("Select a Layer")
    layer_option = st.selectbox(
        "Select an option", list(LAYERS_CONFIG.keys()), index=0, key="layer_option"
    )
    layer_config = LAYERS_CONFIG[layer_option]

    # Extract relevant information from layer_config
    layer_cls = layer_config["layer_cls"]
    layer_args = layer_config["layer_args"]
    control_args = layer_config["control_args"]
    box_format = layer_args["bounding_box_format"]

    # Set control arguments
    layer_args = set_control_args(control_args, layer_args)

    # Instantiate layer
    layer = layer_cls(**layer_args)

    return layer, box_format


def display_editable_table(boxes):
    column_config = {
        "x": st.column_config.NumberColumn(default=0, format="%d"),
        "y": st.column_config.NumberColumn(default=0, format="%d"),
        "w": st.column_config.NumberColumn(default=0, format="%d"),
        "h": st.column_config.NumberColumn(default=0, format="%d"),
    }

    boxes = st.data_editor(
        boxes,
        num_rows="dynamic",
        column_config=column_config,
        hide_index=True,
    )

    np_boxes = np.array(boxes.values.tolist())
    return np_boxes


def display_img_with_bbox(image, bbox, layer, box_format="xywh"):
    images, aug_image = Preprocessing(layer, image, box_format=box_format, boxes=bbox)
    col1, _, col3 = st.columns([0.45, 0.1, 0.45], gap="large")
    with col1:
        st.subheader("Input Image with bbox")
        st.image(images[0], use_column_width=True)
    with col3:
        st.subheader("Output Image with bbox")
        st.image(aug_image, use_column_width=True)


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


def bbox():
    image, bbox = image_dropdown()
    np_boxes = display_editable_table(bbox)
    layer, box_format = select_layer_bbox_aug()
    return layer, image, np_boxes, box_format
