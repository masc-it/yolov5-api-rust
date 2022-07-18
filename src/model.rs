use opencv::{
    core::{self, MatTraitConst, MatTrait, MatExprTraitConst},
    dnn::{self, NetTraitConst, NetTrait}
};
use std::{fs::File, io::{BufReader}, error::Error};
use serde::{Deserialize, Serialize};
use std::os::raw::c_void;

#[derive(Serialize, Deserialize)]
pub struct BoxDetection {

    pub xmin: i32,
    pub ymin: i32,
    pub xmax: i32,
    pub ymax: i32,

    pub class: i32,
    pub conf: f32

}

#[derive(Serialize, Deserialize)]
pub struct Detections {

    pub detections: Vec<BoxDetection>
}

/// Defines model information.
/// - ONNX model absolute path
/// - array of class names
/// - model input size
#[derive(Deserialize)]
pub struct ModelConfig {

    pub model_path : String,
    pub class_names : Vec<String>,
    pub input_size: i32
}

/// Contains the instantiated model itself and its configuration.
pub struct Model {

    pub model: dnn::Net,
    pub model_config: ModelConfig
}

/// Contains information about original input image and effective size fed into the model.
pub struct MatInfo {

    width: f32,
    height: f32,
    scaled_size: f32
}

/// Run detection on input image.
pub fn detect(model_data: &mut Model, img: &core::Mat, conf_thresh: f32, nms_thresh: f32) -> opencv::Result<Detections> {
    
    let model = &mut model_data.model;

    let model_config = &mut model_data.model_config;

    let mat_info = MatInfo{
        width: img.cols() as f32,
        height: img.rows() as f32,

        scaled_size: model_config.input_size as f32
    };

    let padded_mat = prepare_input(&img).unwrap();

    // dnn blob
    let blob = opencv::dnn::blob_from_image(&padded_mat, 1.0 / 255.0, opencv::core::Size_{width: model_config.input_size, height: model_config.input_size}, core::Scalar::new(0f64,0f64,0f64,0f64), true, false, core::CV_32F)?;

    let out_layer_names = model.get_unconnected_out_layers_names()?;

    let mut outs : opencv::core::Vector<core::Mat> = opencv::core::Vector::default();
    model.set_input(&blob, "", 1.0, core::Scalar::default())?;
    
    model.forward(&mut outs, &out_layer_names)?;

    let detections = post_process(&outs, &mat_info, conf_thresh, nms_thresh)?;
    
    Ok(detections)
}

/// Prepare an image as a squared matrix.
fn prepare_input(img: &core::Mat) -> opencv::Result<core::Mat> {

    let width = img.cols();
    let height = img.rows();

    let _max = std::cmp::max(width, height);

    let mut result = opencv::core::Mat::zeros(_max, _max, opencv::core::CV_8UC3).unwrap().to_mat().unwrap();

    img.copy_to(&mut result)?;

    Ok(result)

}

/// Process predictions and apply NMS.
fn post_process(outs: &core::Vector<core::Mat>, mat_info: &MatInfo, conf_thresh: f32, nms_thresh: f32 ) -> opencv::Result<Detections>{

    let mut det = outs.get(0)?;

    let rows = *det.mat_size().get(1).unwrap();
    let cols = *det.mat_size().get(2).unwrap();
    
    let mut boxes: core::Vector<opencv::core::Rect> = core::Vector::default();
    let mut scores: core::Vector<f32> = core::Vector::default();

    let mut indices: core::Vector<i32> = core::Vector::default();

    let mut class_index_list: core::Vector<i32> = core::Vector::default();

    let x_factor = mat_info.width / mat_info.scaled_size;
    let y_factor = mat_info.height / mat_info.scaled_size;

    unsafe {
      
        let data = det.ptr_mut(0)?.cast::<c_void>();

        // safe alternative needed..
        let m = core::Mat::new_rows_cols_with_data(rows, cols, core::CV_32F, data, core::Mat_AUTO_STEP )?;
        
        for r in 0..m.rows() {

            let cx: &f32 = m.at_2d::<f32>(r, 0)?;
            let cy: &f32 = m.at_2d::<f32>(r, 1)?;
            let w: &f32 = m.at_2d::<f32>(r, 2)?;
            let h: &f32 = m.at_2d::<f32>(r, 3)?;
            let sc: &f32 = m.at_2d::<f32>(r, 4)?;
            
            let score = *sc as f64;

            if score < conf_thresh.into() {
                continue;
            }
            let confs = m.row(r)?.col_range( &core::Range::new(5, m.row(r)?.cols())?)?;
            
            let c = (confs * score).into_result()?.to_mat()?;
            
            let mut min_val = Some(0f64);
            let mut max_val = Some(0f64);

            let mut min_loc  = Some(core::Point::default());
            let mut max_loc  = Some(core::Point::default());
            let mut idk = core::no_array();

            // find predicted class with highest confidence
            core::min_max_loc(&c, min_val.as_mut(), max_val.as_mut(), min_loc.as_mut(), max_loc.as_mut(), &mut idk)?;
            
            scores.push(max_val.unwrap() as f32);
            boxes.push( core::Rect{
                x: (((*cx) - (*w) / 2.0) * x_factor).round() as i32, 
                y: (((*cy) - (*h) / 2.0) * y_factor).round() as i32, 
                width: (*w * x_factor).round() as i32, 
                height: (*h * y_factor).round() as i32
            } );
            indices.push(r);

            class_index_list.push(max_loc.unwrap().x);

        }

    }

    // Run NMS.
    dnn::nms_boxes(&boxes, &scores, conf_thresh, nms_thresh, &mut indices, 1.0, 0)?;

    let mut final_boxes : Vec<BoxDetection> = Vec::default();
    
    for i in &indices {

        let indx = i as usize;

        let class = class_index_list.get(indx)?;
        
        let rect = boxes.get(indx)?;

        let bbox = BoxDetection{
            xmin: rect.x,
            ymin: rect.y,
            xmax: rect.x + rect.width,
            ymax: rect.y + rect.height,
            conf: scores.get(indx)?,
            class: class
        };

        final_boxes.push(bbox);
    }

    Ok(Detections{detections: final_boxes})

}


/// Draw predicted bounding boxes.
pub fn draw_predictions(img: &mut core::Mat, detections: &Detections) -> opencv::Result<Vec<u8>> {

    let boxes = &detections.detections;
    for i in 0..boxes.len() {

        let bbox = &boxes[i];
        let rect = opencv::core::Rect::new(bbox.xmin, bbox.ymin, bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin);
        
        // TODO draw labels
        //let label = "A";

        let color = core::Scalar::all(0.0);

        opencv::imgproc::rectangle(img, rect, color, 1, opencv::imgproc::LINE_8, 0)?;
    }

    let mut out_vector :core::Vector<u8>  = core::Vector::default();
    opencv::imgcodecs::imencode(".jpg", img, &mut out_vector, &core::Vector::default()).unwrap();

    Ok(out_vector.to_vec())
}

/// Load model configuration.
/// See ModelConfig.
pub fn load_model_from_config() -> Result<ModelConfig, Box<dyn Error>>{

    let file = File::open("data/config.json")?;
    let reader = BufReader::new(file);

    let j : ModelConfig = serde_json::from_reader(reader)?;

    println!("{model_path}", model_path=j.model_path);

    Ok(j)
}

/// Porting of letterbox padding strategy used to prepare the input image. 
/// 
/// See: https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py#L91
fn letterbox( img: &core::Mat, new_shape: core::Size, scale_up: bool) -> opencv::Result<core::Mat> {

    let width = img.cols() as f32;
    let height = img.rows() as f32;

    let new_width = new_shape.width as f32;
    let new_height = new_shape.height as f32;
    let mut r = f32::min(new_width / width, new_height / height );

    if !scale_up {
        r =f32::min(r, 1.0);
    }

    let new_unpad_w = (width * r).round() as i32;
    let new_unpad_h = (height * r).round() as i32;

    let dw = (new_shape.width - new_unpad_w) / 2;
    let dh = (new_shape.height - new_unpad_h) / 2;

    let mut dst = core::Mat::default();
    opencv::imgproc::resize(&img, &mut dst, core::Size_{width: new_unpad_w, height: new_unpad_h}, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;

    let top =  (dh as f32 - 0.1).round() as i32;
    let bottom =  (dh as f32 + 0.1).round() as i32;
    let left =  (dw as f32 - 0.1).round() as i32;
    let right =  (dw as f32 + 0.1).round() as i32;

    let mut final_mat = core::Mat::default();
    opencv::core::copy_make_border(&dst, &mut final_mat, top, bottom, left, right, opencv::core::BORDER_CONSTANT, opencv::core::Scalar::new(114.0, 114.0, 114.0, 114.0))?;
    
    //let params: core::Vector<i32> = core::Vector::default();
    
    //opencv::imgcodecs::imwrite("padded.jpg", &final_mat, &params)?;
    
    Ok(final_mat)
}