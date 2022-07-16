use opencv::{
    core::{self, Size, ElemMul},
    highgui,
    prelude::*,
    videoio,
    dnn,
    imgcodecs
};

fn run_orig() -> opencv::Result<()> {
    let window = "video capture";
    highgui::named_window(window, 1)?;
    #[cfg(feature = "opencv-32")]
    let mut cam = videoio::VideoCapture::new_default(0)?;  // 0 is the default camera
    #[cfg(not(feature = "opencv-32"))]
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;  // 0 is the default camera
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    let mut frame = core::Mat::default();
    loop {
        cam.read(&mut frame)?;
        if frame.size()?.width > 0 {
            highgui::imshow(window, &mut frame)?;
        }
        let key = highgui::wait_key(1)?;
        if key > 0 && key != 255 {
            break;
        }
    }
    Ok(())
}

struct PadInfo {

    mat: core::Mat,
    top: i32,
    left: i32

}

fn letterbox( img: core::Mat, new_shape: core::Size, scale_up: bool) -> opencv::Result<PadInfo> {


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

    println!("dst");
    let mw = dst.cols();

    let mh = dst.rows();
    
    println!("{mh} - {mw}");

    let top =  (dh as f32 - 0.1).round() as i32;
    let bottom =  (dh as f32 + 0.1).round() as i32;
    let left =  (dw as f32 - 0.1).round() as i32;
    let right =  (dw as f32 + 0.1).round() as i32;

    let mut final_mat = core::Mat::default();
    opencv::core::copy_make_border(&dst, &mut final_mat, top, bottom, left, right, opencv::core::BORDER_CONSTANT, opencv::core::Scalar::new(114.0, 114.0, 114.0, 114.0))?;
    
    let params: core::Vector<i32> = core::Vector::default();
    
    //opencv::imgcodecs::imwrite("lol.jpg", &final_mat, &params)?;
    println!("final");
    let mw = final_mat.cols();

    let mh = final_mat.rows();
    
    println!("{mh} - {mw}");
    
    Ok(PadInfo{mat: final_mat, top: top, left: left})
}
use std::os::raw::c_void;

struct DetectionOuput {

    boxes: core::Vector<opencv::core::Rect>,
    scores: core::Vector<f32>,

    indices: core::Vector<i32>,

    class_index_list: Vec<i32>
}

fn post_process(img: &core::Mat, mut outs: core::Vector<core::Mat>, conf_thresh: f32, nms_thresh: f32 ) -> opencv::Result<(DetectionOuput)>{

    
    let mut det = outs.get(0)?;

    let mut boxes: core::Vector<opencv::core::Rect> = core::Vector::new();
    let mut scores: core::Vector<f32> = core::Vector::new();

    let mut indices: core::Vector<i32> = core::Vector::new();

    let mut class_index_list: core::Vector<i32> = core::Vector::new();

    unsafe {
       /*  let s = det.size()?;
        println!("unsafe");
        let h = s.height;
        let w = s.width;

        println!("{h} - {w}"); */
        let data = det.ptr_mut(0)?.cast::<c_void>();

        let m = core::Mat::new_rows_cols_with_data(102000, 41, core::CV_32F, data, core::Mat_AUTO_STEP )?; // std::mem::size_of::<f32>()* 41
        
        let h = m.rows();
        let w = m.cols();

        println!("{h} - {w}");
        let ones = core::Mat::ones(1, 36, core::CV_32F)?;
        for r in 0..m.rows() {

            //println!("begin");
            let cx: &f32 = m.at_2d::<f32>(r, 0)?;
            let cy: &f32 = m.at_2d::<f32>(r, 1)?;
            let w: &f32 = m.at_2d::<f32>(r, 2)?;
            let h: &f32 = m.at_2d::<f32>(r, 3)?;
            let sc: &f32 = m.at_2d::<f32>(r, 4)?;
            
          

            let score = *sc as f64;
            //println!("score");
            let confs = m.row(r)?.col_range( &core::Range::new(5, m.row(r)?.cols())?)?;
            
            //let c = (confs * score).into_result()?.to_mat()?;

            
            let c = confs.mul(&ones, score)?;
        
            let mut min_val = Some(0f64);
            let mut max_val = Some(0f64);

            let mut min_loc  = Some(core::Point::default());
            let mut max_loc  = Some(core::Point::default());
            let mut idk = core::no_array();
            //println!("ok 1");
            core::min_max_loc(&c, min_val.as_mut(), max_val.as_mut(), min_loc.as_mut(), max_loc.as_mut(), &mut idk)?;
            scores.push(max_val.unwrap() as f32 );
            //println!("ok 2");
            
            boxes.push( core::Rect{x: ((*cx) - (*w) / 2.0).round() as i32, y: ((*cy) - (*h) / 2.0).round() as i32, width: *w as i32, height: *h as i32} );
            
            indices.push(r);
            
            class_index_list.push(max_loc.unwrap().x);
            //println!("ok 3");

           // println!("ok nms");
        }

    }
    dnn::nms_boxes(&boxes, &scores, 0.5, 0.45, &mut indices, 1.0, 0)?;
    let mut indxs : Vec<i32> = Vec::new();
    for i in &indices {
        indxs.push(class_index_list.get(i as usize)?);
    }
    //let m = core::Mat(dets[0].size()?.height, dets[0].size()?.width, core::CV_32F, outs.get(0)?.as_raw(), opencv::core::Mat_AUTO_STEP);

    

    /* for i in 0..outs.len() {
        
        let v : f32 = *mat.at(i as i32)?;
        println!("{v}");
    } */
    Ok(DetectionOuput{
        boxes: boxes,
        scores: scores,
        indices: indices,
        class_index_list: indxs
    })

}

fn run() -> opencv::Result<()> {
    
    let model_path = "D:\\Download\\letters_best1507b.onnx";
    let input_size = 1280;

    let img_path = "D:\\Documenti\\datasets\\letters_detection\\val\\25c.jpg";

    let mut model = dnn::read_net_from_onnx(model_path)?;
    
    //model.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    println!("Loaded model");

    let mat = imgcodecs::imread(img_path, opencv::imgcodecs::IMREAD_COLOR)?;

    /* let mw = mat.cols();

    let mh = mat.rows();
    
    println!("{mh} - {mw}"); */

    println!("mat copy");
    let mat_copy = mat.clone();

    /* let mcw = mat_copy.cols();

    let mch = mat_copy.rows();
    
    println!("{mch} - {mcw}");
 */
    // letterbox

    let pad_info = letterbox(mat_copy, core::Size::new(1280, 1280), true)?;

    let padded_mat = pad_info.mat.clone();

    println!("Pad image");
    let mw = padded_mat.cols();

    let mh = padded_mat.rows();
    
    println!("{mh} - {mw}");
    // dnn blob

    let blob = opencv::dnn::blob_from_image(&padded_mat, 1.0 / 255.0, opencv::core::Size_{width: 1280, height: 1280}, core::Scalar::new(0f64,0f64,0f64,0f64), true, false, core::CV_32F)?;

    println!("Blob");

    let out_layer_names = model.get_unconnected_out_layers_names()?;

    
    let mut outs : opencv::core::Vector<core::Mat> = opencv::core::Vector::default();
    model.set_input(&blob, "", 1.0, core::Scalar::default())?;
    
    model.forward(&mut outs, &out_layer_names)?;


    let detection_output = post_process(&padded_mat, outs,0.5, 0.5)?;

    draw_predictions(pad_info.mat.clone(), detection_output)?;
    println!("Forward pass OK");
    
    Ok(())
}

fn draw_predictions(mut img: core::Mat, detection_output: DetectionOuput) -> opencv::Result<()> {

    let boxes = detection_output.boxes;
    let scores = detection_output.scores;
    let indices = detection_output.indices;
    let class_index_list = detection_output.class_index_list;

    let l = indices.len();
    println!("{l}");
    for i in 0..indices.len() {
        let rect = boxes.get(indices.get(i)? as usize)?;

        let label = "A";

        let color = core::Scalar::all(0.0);

        opencv::imgproc::rectangle(&mut img, rect, color, 1, opencv::imgproc::LINE_8, 0)?;
    }

    opencv::imgcodecs::imwrite("boxes.jpg", &img, &core::Vector::default())?;
    Ok(())
}

fn main() {
    run().unwrap()
}
