import keras_cv
import streamlit as st


LAYERS_CONFIG = {
    "AutoContrast": {
        "layer_cls": keras_cv.layers.AutoContrast,
        "layer_args": {
            "value_range": (0, 255),
        },
        "control_args": {},
        "is_compatible_with_bbox": False,
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
        "is_compatible_with_bbox": False,
    },
    "ChannelShuffle": {
        "layer_cls": keras_cv.layers.ChannelShuffle,
        "layer_args": {
            "groups": 3,
        },
        "control_args": {
            "groups": [1, 3],
        },
        "is_compatible_with_bbox": False,
    },
    "GridMask": {
        "layer_cls": keras_cv.layers.GridMask,
        "layer_args": {
            "ratio_factor": (0, 0.5),
            "rotation_factor": (0.0, 0.0),
            "fill_mode": "constant",
        },
        "control_args": {
            "ratio_factor": (0.1, 1.0),
            "rotation_factor": (-90.0, 90.0),
            "fill_mode": ["constant", "gaussian_noise"],
        },
        "is_compatible_with_bbox": False,
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
        "is_compatible_with_bbox": False,
    },
    "RandomColorDegeneration": {
        "layer_cls": keras_cv.layers.RandomColorDegeneration,
        "layer_args": {
            "factor": 0.0,
        },
        "control_args": {
            "factor": [0.0, 1.0],
        },
        "is_compatible_with_bbox": False,
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
        "is_compatible_with_bbox": False,
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
        "is_compatible_with_bbox": False,
    },
    "RandomSaturation": {
        "layer_cls": keras_cv.layers.RandomSaturation,
        "layer_args": {
            "factor": 0.0,
        },
        "control_args": {
            "factor": [0.0, 1.0],
        },
        "is_compatible_with_bbox": False,
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
        "is_compatible_with_bbox": False,
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
        "is_compatible_with_bbox": False,
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
        "is_compatible_with_bbox": False,
    },
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
        "is_compatible_with_bbox": True,
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
        "is_compatible_with_bbox": True,
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
        "is_compatible_with_bbox": True,
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
        "is_compatible_with_bbox": True,
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
        "is_compatible_with_bbox": True,
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
        "is_compatible_with_bbox": False,
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
        "is_compatible_with_bbox": True,
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
        "is_compatible_with_bbox": True,
    },
    "Grayscale": {
        "layer_cls": keras_cv.layers.Grayscale,
        "layer_args": {
            "output_channels": 1,
        },
        "control_args": {},
        "is_compatible_with_bbox": False,
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
        "is_compatible_with_bbox": False,
    },
}