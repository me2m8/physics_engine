
use physics_engine::{ctx::Ctx, state::*};

#[tokio::main]
async fn main() {
    let event_loop = winit::event_loop::EventLoopBuilder::new().build().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_title("Cum: the gamme")
        .with_maximized(true)
        .build(&event_loop).unwrap();

    let mut state = State::new(&window).await;
    let mut ctx = Ctx::new(&state);

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
                winit::event::WindowEvent::RedrawRequested => {
                    // Do some rendering and stuff
                    update(&state, &mut ctx);
                    render(&state, &ctx);
                }
                winit::event::WindowEvent::Resized(new_size) => {
                    state.resize(new_size);
                    ctx.camera.scale_with_view(&state, new_size);
                }
                _ => {}
            }
        },
        winit::event::Event::AboutToWait => state.window().request_redraw(),
        _ => {}
    });
}

fn render(state: &State, ctx: &Ctx) {
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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            }
        );

        render_pass.set_pipeline(&ctx.pipeline);
        render_pass.set_bind_group(0, &ctx.camera.bind_group, &[]);

        render_pass.set_vertex_buffer(0, ctx.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, ctx.instance_buffer.slice(..));
        render_pass.set_index_buffer(ctx.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        render_pass.draw_indexed(0..ctx.num_indicies, 0, 0..ctx.num_instances);
    }

    state.queue().submit(std::iter::once(encoder.finish()));
    output.present();
}

fn update(state: &State, ctx: &mut Ctx) {
    ctx.physics_process(state);
}