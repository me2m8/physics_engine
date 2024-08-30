
use cgmath::Vector2;
use physics_engine::{ctx::Ctx, instances::Renderer as _, state::*, *};

use tracing::Level;

#[tokio::main]
async fn main() {
     // tracing_subscriber::fmt()
     //     .with_max_level(Level::TRACE)
     //     .with_timer(tracing_subscriber::fmt::time::uptime())
     //     .init();

    let event_loop = winit::event_loop::EventLoopBuilder::new().build().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_title("Cum: the gamme")
        .with_active(true)
        .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
        .build(&event_loop).unwrap();


    let mut state = State::new(&window).await;
    let mut ctx = Ctx::new(&state, Vector2::new(WINDOW_WIDTH, WINDOW_HEIGHT));

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
                winit::event::WindowEvent::KeyboardInput { event, .. } => {
                    ctx.camera.process_input(event);
                }
                winit::event::WindowEvent::RedrawRequested => {
                    ctx.update_dt();

                    // Do some rendering and stuff
                    update(&state, &mut ctx);
                    render(&mut state, &ctx);
                }
                winit::event::WindowEvent::Resized(new_size) => {
                    println!("Resizing to: {new_size:#?}");

                    state.resize(new_size);
                    ctx.camera.scale_with_view(&state, new_size);

                    let half_bw = BORDER_WIDTH / 2.;
                    let half_width = new_size.width as f32;
                    let half_height = new_size.height as f32;

                    let new_border = [
                        Vector2::new(half_width - half_bw, half_height - half_bw),
                        Vector2::new(-(half_width - half_bw), half_height - half_bw),
                        Vector2::new(-(half_width - half_bw), -(half_height - half_bw)),
                        Vector2::new(half_width - half_bw, -(half_height - half_bw)),
                    ];

                    ctx.border.set_vertices(new_border);
                    ctx.polygon_frame_render.update_buffers(&state, &[ctx.border.clone()]);
                }
                _ => {}
            }
        },
        winit::event::Event::AboutToWait => state.window().request_redraw(),
        _ => {}
    });
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
                            r: 0.2,
                            g: 0.2,
                            b: 0.2,
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

        ctx.circle_render.render(&mut render_pass);
        ctx.polygon_frame_render.render(&mut render_pass);
    }

    state.queue().submit(std::iter::once(encoder.finish()));
    output.present();

    let mut fps = 0.0;
    let secs = ctx.dt().as_secs_f32();
    if secs != 0.0 {
        fps = 1.0 / secs;
    }

    log::log!(log::Level::Info, "FPS: {fps:?}");
}

fn update(state: &State, ctx: &mut Ctx) {
    ctx.physics_process(state);
}
