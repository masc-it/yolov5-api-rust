use std::{fs::File, io::{BufReader}, error::Error};

use opencv::{
    core::{self},
    prelude::*,
    dnn,
    imgcodecs
};
use serde::{Deserialize};
mod model;

#[derive(Deserialize, Debug)]
struct ModelConfig {

    pub model_path : String,
    pub class_names : Vec<String>,
    pub input_size: i32
}

fn load_model_config() -> Result<ModelConfig, Box<dyn Error>>{

    let file = File::open("data/config.json")?;
    let reader = BufReader::new(file);

    let j : ModelConfig = serde_json::from_reader(reader)?;

    println!("{model_path}", model_path=j.model_path);

    Ok(j)
}

fn run() -> Result<(), Box<dyn Error>> {
    
    let model_config = load_model_config()?;

    // TODO load model

    detect()?;

    Ok(())
}

fn detect() -> opencv::Result<()> {
    
    let model_path = "D:\\Download\\letters_best1507b.onnx";
    let input_size = 1280;

    let img_path = "D:\\Documenti\\datasets\\letters_detection\\val\\25c.jpg";

    let mut model = dnn::read_net_from_onnx(model_path)?;
    
    model.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    println!("Loaded model");

    let mat = imgcodecs::imread(img_path, opencv::imgcodecs::IMREAD_COLOR)?;

    println!("mat copy");
    let mat_copy = mat.clone();

    // letterbox

    let pad_info = model::letterbox(&mat_copy, core::Size::new(1280, 1280), true)?;

    let padded_mat = pad_info.mat.clone();

    // dnn blob

    let blob = opencv::dnn::blob_from_image(&padded_mat, 1.0 / 255.0, opencv::core::Size_{width: 1280, height: 1280}, core::Scalar::new(0f64,0f64,0f64,0f64), true, false, core::CV_32F)?;

    println!("Blob");

    let out_layer_names = model.get_unconnected_out_layers_names()?;

    
    let mut outs : opencv::core::Vector<core::Mat> = opencv::core::Vector::default();
    model.set_input(&blob, "", 1.0, core::Scalar::default())?;
    
    model.forward(&mut outs, &out_layer_names)?;

    let detection_output = model::post_process(&padded_mat, &outs,0.5, 0.5)?;

    model::draw_predictions(&mut pad_info.mat.clone(), &detection_output)?;
    println!("Forward pass OK");
    
    Ok(())
}


fn main() {
    run().unwrap()
}
