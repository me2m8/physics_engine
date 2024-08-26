#[macro_use]
extern crate glium;
use self::glium::{
    winit::{
        event::{ElementState, KeyEvent},
        keyboard::{KeyCode, PhysicalKey},
    },
    Surface,
};

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}
implement_vertex!(Vertex, position);

#[rustfmt::skip]
const SHAPE: &[Vertex] = &[
    Vertex { position: [-0.5, -0.5]  },
    Vertex { position: [ 0.0,  0.5]  },
    Vertex { position: [ 0.5, -0.25] },
];

fn main() {
    let event_loop = glium::winit::event_loop::EventLoop::builder()
        .build()
        .expect("event loop should be built");
    let (_window, display) = glium::backend::glutin::SimpleWindowBuilder::new().build(&event_loop);

    let vertex_buffer = glium::VertexBuffer::new(&display, SHAPE).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let program = glium::Program::from_source(
        &display,
        include_str!("shader.vert"),
        include_str!("shader.frag"),
        None,
    )
    .unwrap();

    let mut t: f32 = 0.0;

    let _ = event_loop.run(move |event, window_target| match event {
        glium::winit::event::Event::WindowEvent { event, .. } => match event {
            glium::winit::event::WindowEvent::CloseRequested
            | glium::winit::event::WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => window_target.exit(),
            glium::winit::event::WindowEvent::RedrawRequested => {
                t += 0.1;

                let offset = t.sin() * 0.5;
                let scale = (t + 0.9).sin() * 0.5 + 0.5;

                let mut target = display.draw();
                target.clear_color(0.1, 0.2, 0.3, 1.0);
                target
                    .draw(
                        &vertex_buffer,
                        indices,
                        &program,
                        &uniform! { x: offset, s: scale },
                        &Default::default(),
                    )
                    .unwrap();
                target.finish().unwrap();
            }

            _ => {}
        },
        glium::winit::event::Event::AboutToWait => {
            _window.request_redraw();
        }
        _ => {}
    });
}
