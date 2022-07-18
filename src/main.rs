

use std::error::Error;
use std::sync::Mutex;

use model::Model;
use opencv::{
    dnn
};
use opencv::prelude::{NetTrait};
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};

mod model;


fn load_model() -> Result<Model, Box<dyn Error>> {

    let model_config = model::load_model_from_config()?;

    let mut model = dnn::read_net_from_onnx(&model_config.model_path)?;
    model.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;

    Ok(model::Model{model, model_config: model_config})

} 

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Hello world!")
}

#[post("/predict")]
async fn predict(data: web::Data<AppState>, req_body: web::Bytes) -> impl Responder {

    let mut model = &mut *data.model.lock().unwrap();

    let img: Vec<u8> = req_body.to_vec();

    let img_vec:  opencv::core::Vector<u8> = opencv::core::Vector::from_iter(img);
    let mat = opencv::imgcodecs::imdecode(&img_vec, opencv::imgcodecs::IMREAD_UNCHANGED).unwrap();

    model::detect(&mut model,&mat).unwrap();
    
    HttpResponse::Ok().body("ok")
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
    let model = load_model()?;
    
    main_api(model).unwrap();

    Ok(())
}
