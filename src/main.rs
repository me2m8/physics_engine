
use physics_engine::state::*;

fn main() {
    let event_loop = winit::event_loop::EventLoopBuilder::new().build().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_title("Cum: the gamme")
        .with_active(true)
        .build(&event_loop).unwrap();

    let _ = event_loop.run(move |event, control_flow| match event {
        winit::event::Event::WindowEvent { window_id, event } if window_id == window.id() => {
            match event {
                winit::event::WindowEvent::CloseRequested
                | winit::event::WindowEvent::KeyboardInput { 
                    event: winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                        state: winit::event::ElementState::Pressed,
                        ..
                    }, 
                    .. 
                } => control_flow.exit(),
                _ => {}
            }
        },
        _ => {}
    });
}
