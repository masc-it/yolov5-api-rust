# YoloV5-API [WIP]

API to run inferences with YoloV5 models. Written in Rust, based on OpenCV 4.5.5

If you need a C++ version, check my [C++ Yolov5-API](https://github.com/masc-it/yolov5-api-cpp)

## Requirements

- [OpenCV 4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5) installed on your system
  - Follow [Rust opencv README](https://github.com/twistedfall/opencv-rust)

## Model config

**Data** directory must contain your config.json

**config.json** defines:
- ONNX absolute model path
- input size (640 default)
- array of class names

A dummy example is available in the _data/_ folder


## Docker

Soon.

## Build

Development:

    cargo build

Release:

    cargo build --release

## Run

    cargo run

For Windows users: Assure to have _opencv_world455.dll_ in your exe directory.

# Endpoints

## /predict [POST]

### Body
- Image bytes (binary in Postman)

### Headers
- X-Confidence-Thresh
  - default 0.5
- X-NMS-Thresh
  - default 0.45
- X-Return
  - image_with_boxes
    - A JPEG image with drawn predictions
  - json (default)
    - A json array containing predictions. Each object defines: xmin, ymin, xmax, ymax, conf, class_name