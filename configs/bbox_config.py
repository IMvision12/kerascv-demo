import keras_cv
import streamlit as st


LAYERS_CONFIG = {
    "RandomShear": {
        "layer_cls": keras_cv.layers.RandomShear,
        "layer_args": {
            "x_factor": 0.0,
            "y_factor": 0.0,
            "interpolation": "bilinear",
            "fill_mode": "reflect",
            "fill_value": 0.0,
            "bounding_box_format": "xywh",
        },
        "control_args": {
            "x_factor": (0.0, 1.0),
            "y_factor": (0.0, 1.0),
            "interpolation": ["bilinear", "nearest"],
            "fill_mode": ["reflect", "wrap", "constant", "nearest"],
            "bounding_box_format": ["xyxy", "xywh"],
        },
    },
    "RandomCrop": {
        "layer_cls": keras_cv.layers.RandomCrop,
        "layer_args": {
            "height": 150,
            "width": 150,
            "bounding_box_format": "xywh",
        },
        "control_args": {
            "height": (10, 512),
            "width": (10, 512),
            "bounding_box_format": ["xyxy", "xywh"],
        },
    },
    "RandomRotation": {
        "layer_cls": keras_cv.layers.RandomRotation,
        "layer_args": {
            "factor": 0.0,
            "fill_mode": "reflect",
            "interpolation": "nearest",
            "bounding_box_format": "xywh",
        },
        "control_args": {
            "factor": (0.1, 1.0),
            "fill_mode": ["reflect", "constant", "wrap", "nearest"],
            "interpolation": ["nearest", "bilinear"],
            "bounding_box_format": ["xyxy", "xywh"],
        },
    },
    "RandomTranslation": {
        "layer_cls": keras_cv.layers.RandomTranslation,
        "layer_args": {
            "height_factor": 0.0,
            "width_factor": 0.0,
            "fill_mode": "reflect",
            "interpolation": "nearest",
            "bounding_box_format": "xywh",
        },
        "control_args": {
            "height_factor": (0.1, 1.0),
            "width_factor": (0.1, 1.0),
            "fill_mode": ["reflect", "constant", "wrap", "nearest"],
            "interpolation": ["nearest", "bilinear"],
            "bounding_box_format": ["xyxy", "xywh"],
        },
    },
    "RandomFlip": {
        "layer_cls": keras_cv.layers.RandomFlip,
        "layer_args": {
            "mode": "horizontal",
            "rate": 0.5,
            "bounding_box_format": "xywh",
        },
        "control_args": {
            "mode": ["horizontal", "vertical"],
            "bounding_box_format": ["xyxy", "xywh"],
        },
    },
    "RandAugment": {
        "layer_cls": keras_cv.layers.RandAugment,
        "layer_args": {
            "value_range": (0, 255),
            "augmentations_per_image": 2,
            "magnitude": 10.0,
            "magnitude_stddev": 0.0,
            "rate": 0.9090909090909091,
            "geometric": True,
            "bounding_box_format": "xywh",
        },
        "control_args": {
            "augmentations_per_image": [1, 3],
            "magnitude": [0.0, 30.0],
            "geometric": [True, False],
            "bounding_box_format": ["xyxy", "xywh"],
        },
    },
    "Resizing": {
        "layer_cls": keras_cv.layers.Resizing,
        "layer_args": {
            "height": 150,
            "width": 150,
            "interpolation": "bilinear",
            "bounding_box_format": "xywh",
        },
        "control_args": {
            "height": [10, 512],
            "width": [10, 512],
            "interpolation": ["bilinear", "nearest", "bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic"],
            "bounding_box_format": ["xyxy", "xywh"],
        },
    },
}