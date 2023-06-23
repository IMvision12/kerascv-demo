import keras_cv
import streamlit as st


LAYERS_CONFIG = {
    "AutoContrast": {
        "layer_cls": keras_cv.layers.AutoContrast,
        "layer_args": {
            "value_range": (0, 255),
        },
        "control_args": {},
    },
    "AugMix": {
        "layer_cls": keras_cv.layers.AugMix,
        "layer_args": {
            "value_range": (0, 255),
            "severity": [0.01, 0.3],
            "num_chains": 3,
            "chain_depth": [1, 3],
            "alpha": 1.0,
        },
        "control_args": {
            "severity": [0.01, 1.0],
            "num_chains": [1, 5],
            "chain_depth": [1, 5],
            "alpha": [0.01, 2.0],
        },
    },
    "ChannelShuffle": {
        "layer_cls": keras_cv.layers.ChannelShuffle,
        "layer_args": {
            "groups": 3,
        },
        "control_args": {
            "groups": [1, 3],
        },
    },
    "GridMask": {
        "layer_cls": keras_cv.layers.GridMask,
        "layer_args": {
            "ratio_factor": (0.0, 0.5),
            "rotation_factor": (0.0, 0.0),
            "fill_mode": "constant",
        },
        "control_args": {
            "ratio_factor": (0.1, 1.0),
            "rotation_factor": (-90.0, 90.0),
            "fill_mode": ["constant", "gaussian_noise"],
        },
    },
    "RandomChannelShift": {
        "layer_cls": keras_cv.layers.RandomChannelShift,
        "layer_args": {
            "value_range": (0, 255),
            "factor": 0.0,
            "channels": 3,
        },
        "control_args": {
            "factor": [0.1, 1.0],
        },
    },
    "RandomColorDegeneration": {
        "layer_cls": keras_cv.layers.RandomColorDegeneration,
        "layer_args": {
            "factor": 0.0,
        },
        "control_args": {
            "factor": [0.0, 1.0],
        },
    },
    "RandomCutout": {
        "layer_cls": keras_cv.layers.RandomCutout,
        "layer_args": {
            "height_factor": (0.5, 0.5),
            "width_factor": (0.5, 0.5),
            "fill_mode": "constant",
            "fill_value": "constant",
        },
        "control_args": {
            "height_factor": (0.1, 1.0),
            "width_factor": (0.1, 1.0),
            "fill_mode": ["constant", "gaussian_noise"]
        },
    },
    "RandomHue": {
        "layer_cls": keras_cv.layers.RandomHue,
        "layer_args": {
            "factor": 0.0,
            "value_range": (0, 255),
        },
        "control_args": {
            "factor": [0.0, 1.0],
        },
    },
    "RandomSaturation": {
        "layer_cls": keras_cv.layers.RandomSaturation,
        "layer_args": {
            "factor": 0.0,
        },
        "control_args": {
            "factor": [0.0, 1.0],
        },
    },
    "RandomContrast": {
        "layer_cls": keras_cv.layers.RandomContrast,
        "layer_args": {
            "value_range": (0, 255),
            "factor": 0.0,
        },
        "control_args": {
            "factor": [0.0, 1.0],
        },
    },
    "RandomBrightness": {
        "layer_cls": keras_cv.layers.RandomBrightness,
        "layer_args": {
            "value_range": (0, 255),
            "factor": 0.0,
        },
        "control_args": {
            "factor": [0.0, 1.0],
        },
    },
    "RandomSharpness": {
        "layer_cls": keras_cv.layers.RandomSharpness,
        "layer_args": {
            "value_range": (0, 255),
            "factor": 0.0,
        },
        "control_args": {
            "factor": (0.0, 1.0),
        },
    },
    "RandomShear": {
        "layer_cls": keras_cv.layers.RandomShear,
        "layer_args": {
            "x_factor": 0.0,
            "y_factor": 0.0,
            "interpolation": "bilinear",
            "fill_mode": "reflect",
            "fill_value": 0.0,
        },
        "control_args": {
            "x_factor": (0.0, 1.0),
            "y_factor": (0.0, 1.0),
            "interpolation": ["bilinear", "nearest"],
            "fill_mode": ["reflect", "wrap", "constant", "nearest"],
        },
    },
    "RandomCrop": {
        "layer_cls": keras_cv.layers.RandomCrop,
        "layer_args": {
            "height": 150,
            "width": 150,
        },
        "control_args": {
            "height": (10, 512),
            "width": (10, 512),
        },
    },
    "RandomRotation": {
        "layer_cls": keras_cv.layers.RandomRotation,
        "layer_args": {
            "factor": 0.0,
            "fill_mode": "reflect",
            "interpolation": "nearest",
        },
        "control_args": {
            "factor": (0.1, 1.0),
            "fill_mode": ["reflect", "constant", "wrap", "nearest"],
            "interpolation": ["nearest", "bilinear"],
        },
    },
    "RandomTranslation": {
        "layer_cls": keras_cv.layers.RandomTranslation,
        "layer_args": {
            "height_factor": 0.0,
            "width_factor": 0.0,
            "fill_mode": "reflect",
            "interpolation": "nearest",
        },
        "control_args": {
            "height_factor": (0.1, 1.0),
            "width_factor": (0.1, 1.0),
            "fill_mode": ["reflect", "constant", "wrap", "nearest"],
            "interpolation": ["nearest", "bilinear"],
        },
    },
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
    "Solarization": {
        "layer_cls": keras_cv.layers.Solarization,
        "layer_args": {
            "value_range": (0, 255),
            "addition_factor": 0.0,
            "threshold_factor": 0.0,
        },
        "control_args": {
            "addition_factor": (0.0, 1.0),
            "addition_factor": (0.0, 1.0),
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
        },
        "control_args": {
            "augmentations_per_image": [1, 3],
            "magnitude": [0.0, 30.0],
            "geometric": [True, False],
        },
    },
    "Resizing": {
        "layer_cls": keras_cv.layers.Resizing,
        "layer_args": {
            "height": 150,
            "width": 150,
            "interpolation": "bilinear",
        },
        "control_args": {
            "height": [10, 512],
            "width": [10, 512],
            "interpolation": ["bilinear", "nearest", "bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic"],
        },
    },
    "Grayscale": {
        "layer_cls": keras_cv.layers.Grayscale,
        "layer_args": {
            "output_channels": 1,
        },
        "control_args": {},
    },
    "Equalization": {
        "layer_cls": keras_cv.layers.Equalization,
        "layer_args": {
            "value_range": (0, 255),
            "bins": 0,
        },
        "control_args": {
            "bins": [0, 255]
        },
    },
}