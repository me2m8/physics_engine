
use cgmath::Vector2;
use physics_engine::{ctx::Ctx, instances::Renderer as _, state::*, *};

#[tokio::main]
async fn main() {
     // tracing_subscriber::fmt()
     //     .with_max_level(tracing::Level::TRACE)
     //     .with_timer(tracing_subscriber::fmt::time::uptime())
     //     .init();

    let event_loop = winit::event_loop::EventLoopBuilder::new().build().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_title("Cum: the gamme")
        .with_active(true)
        .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(&event_loop).unwrap();


    let mut state = State::new(&window).await;
    let mut ctx = Ctx::new(&state, Vector2::new(VIEWPORT_WIDTH, VIEWPORT_HEIGHT));

    #[allow(clippy::collapsible_match)]
    let _ = event_loop.run(move |event, control_flow| match event {
        winit::event::Event::WindowEvent { window_id, event } if window_id == state.window().id() => {
            match event {
                winit::event::WindowEvent::CloseRequested 
                | winit::event::WindowEvent::KeyboardInput { 
                    event:
                        winit::event::KeyEvent {
                            physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                            state: winit::event::ElementState::Pressed,
                            ..
                        }, 
                    ..
                } => control_flow.exit(),
                winit::event::WindowEvent::KeyboardInput { .. } => {}
                winit::event::WindowEvent::RedrawRequested => {
                    process_frame(&mut state, &mut ctx);
                }
                winit::event::WindowEvent::Resized(new_size) => {
                    println!("Resizing to: {new_size:#?}");

                    state.resize(new_size);
                }
                _ => {}
            }
        },
        winit::event::Event::AboutToWait => state.window().request_redraw(),
        _ => {}
    });
}

fn process_frame(state: &mut State, ctx: &mut Ctx) {
    update(state, ctx);
    render(state, ctx);
}

fn render(state: &mut State, ctx: &Ctx) {

    let mut encoder = state.device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        }
    );

    let output = state.surface().get_current_texture().unwrap();

    let multisample_view = state.multisample_texture().create_view(&wgpu::TextureViewDescriptor::default());
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

    {
        let mut render_pass = encoder.begin_render_pass(
            &wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &multisample_view,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            }
        );

        render_pass.set_bind_group(0, &ctx.camera.bind_group, &[]);

        ctx.circle_renderer.render(&mut render_pass);
        ctx.polygon_frame_render.render(&mut render_pass);
    }

    state.queue().submit(std::iter::once(encoder.finish()));
    output.present();
}

fn update(state: &State, ctx: &mut Ctx) {
    let dt = 1.0 / 75.0;

    ctx.simulation_ctx.update(dt);
    ctx.simulation_ctx.draw_particles(state, &mut ctx.circle_renderer);
}
