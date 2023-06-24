import keras_cv
import streamlit as st


LAYERS_CONFIG = {
    "RandomFlip": {
        "layer_cls": keras_cv.layers.RandomFlip,
        "layer_args": {
            "mode": "horizontal",
            "rate": 0.5,
        },
        "control_args": {
            "mode": ["horizontal", "vertical"],
        },
    },
}