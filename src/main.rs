
use std::time::Instant;
use std::error::Error;
use std::sync::Mutex;

use model::Model;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder, HttpRequest};

mod model;

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Hello world!")
}

#[post("/predict")]
async fn predict(data: web::Data<AppState>, req: HttpRequest, req_body: web::Bytes) -> impl Responder {

    let start = Instant::now();
    let mut model = &mut *data.model.lock().unwrap();

    let img: Vec<u8> = req_body.to_vec();

    let conf_thresh = req.headers().get("X-Confidence-Thresh")
        .map_or_else(|| 0.5, 
            |f| f.to_str().unwrap().parse::<f32>().map_or(0.5, |v| v));


    let nms_thresh = req.headers().get("X-NMS-Thresh")
        .map_or_else(|| 0.5, 
            |f| f.to_str().unwrap().parse::<f32>().map_or(0.5, |v| v));

    let mut return_type = req.headers().get("X-Return")
        .map_or_else(|| "json", |f| f.to_str().unwrap());

    if !return_type.eq("json") && !return_type.eq("img_with_boxes") {
        return_type = "json";
    }
    let img_vec:  opencv::core::Vector<u8> = opencv::core::Vector::from_iter(img);
    let mat = opencv::imgcodecs::imdecode(&img_vec, opencv::imgcodecs::IMREAD_UNCHANGED);
    
    let mut mat = mat.unwrap();

    let detections = model::detect(&mut model,&mat, conf_thresh, nms_thresh);
    
    if detections.is_err() {
        return HttpResponse::Ok()
            .append_header(("Content-Type", "application/json"))
            .body("{\"msg\": \"Invalid image.\"}")
    } 

    let detections = detections.unwrap();
    let duration = start.elapsed();
    if return_type.eq("json") {
        let json_response = serde_json::to_string_pretty(&detections).unwrap();
        HttpResponse::Ok()
            .append_header(("X-Duration", duration.as_millis().to_string()))
            .append_header(("Content-Type", "application/json"))
            .body(json_response)
    } else {

        let img_with_boxes_bytes = model::draw_predictions(&mut mat, &detections, &model.model_config).unwrap();

        HttpResponse::Ok()
            .append_header(("X-Duration", duration.as_millis().to_string()))
            .append_header(("Content-Type", "image/jpeg"))
            .body(img_with_boxes_bytes)
    }
    
    

    
}

struct AppState {

    model: Mutex<Model>
}

#[actix_web::main]
async fn main_api(model: Model) -> std::io::Result<()> {
    
    let data = web::Data::new(AppState {
        model: Mutex::new(model),
    });

    HttpServer::new(move || {
        App::new()
        .app_data(data.clone())
            .service(hello)
            .service(predict)
    })
    .bind(("0.0.0.0", 5000))?
    .run()
    .await
}

fn main() -> Result<(), Box<dyn Error>>{
    let model = model::load_model()?;
    
    main_api(model).unwrap();

    Ok(())
}
