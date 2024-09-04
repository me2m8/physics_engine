
use std::sync::mpsc;

use cgmath::Vector2;
use physics_engine::application::Application;

#[tokio::main]
async fn main() {
    // tracing_subscriber::fmt()
    //     .with_max_level(tracing::Level::TRACE)
    //     .with_timer(tracing_subscriber::fmt::time::uptime())
    //     .init();


    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let (tx, rx) = mpsc::channel();

    let app = Application::new(&event_loop, rx, tx);

    let _ = event_loop.run_app(app);
}
