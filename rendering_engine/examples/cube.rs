use rendering_engine::{state::State, vertex::Vertex};

use wgpu::{include_wgsl, util::DeviceExt};
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

struct Context {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    num_vertices: u32,
}

impl Context {
    fn new(state: &State) -> Self {

        let circle_vertices = create_circle_vertices(20, 0.5, [1.0, 1.0, 1.0]);
        dbg!(&circle_vertices, circle_vertices.len());

        let shader = state.device.create_shader_module(include_wgsl!("cube.wgsl"));

        let pipeline_layout = state.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            }
        );

        let render_pipeline = state.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Line,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 4,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None, 
                cache: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Vertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: state.config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
            }
        );

        let vertex_buffer = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&circle_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            });

        let num_indices = INDICES.len() as u32;
        let num_vertices = circle_vertices.len() as u32;

        Self {
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            num_vertices,
        }
    }
}

#[rustfmt::skip]
const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5, -0.5,  0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 0.5, -0.5,  0.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [ 0.5,  0.5,  0.0], color: [0.0, 0.0, 1.0] },
    Vertex { position: [-0.5,  0.5,  0.0], color: [1.0, 1.0, 1.0] },
];

#[rustfmt::skip]
const INDICES: &[u16] = &[
    // Front
    0, 1, 2, 
    3, 2, 0,
];

#[tokio::main]
async fn main() {
    run().await;
}

async fn run() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    let mut state: State = State::new(&window).await;

    let context = Context::new(&state);

    #[allow(clippy::collapsible_match)]
    let _ = event_loop.run(move |event, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => control_flow.exit(),
            WindowEvent::RedrawRequested => {
                update(&mut state);
                render(&context, &mut state);
            }
            WindowEvent::Resized(new_size) => {
                state.resize(*new_size);
            }
            _ => {}
        },
        Event::AboutToWait => {
            state.window.request_redraw();
        }
        _ => {}
    });
}

fn update(state: &mut State) {}

fn render(ctx: &Context, state: &mut State) {
    let multisample_texture = state.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Multisample Texture"),
        size: wgpu::Extent3d {
            width: state.canvas_size.width,
            height: state.canvas_size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format: state.config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let multisample_view = multisample_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let (output, view) = state.create_view().unwrap();
    let mut encoder = state.create_encoder("Command Encoder");

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        });

        render_pass.set_pipeline(&ctx.render_pipeline);
        render_pass.set_vertex_buffer(0, ctx.vertex_buffer.slice(..));
        render_pass.draw(0..ctx.num_vertices, 0..1);
    }

    state.present_render(output, encoder);
}



fn create_circle_vertices(resolution: u32, radius: f32, color: [f32; 3]) -> Vec<Vertex>{
    let mut vertices: Vec<Vertex> = vec![];
    const TWO_PI: f32 = std::f32::consts::TAU;
    let radial_step = TWO_PI / resolution as f32;

    (0..=resolution).for_each(|i| {
        let x = (i as f32 * radial_step).cos() * radius;
        let y = (i as f32 * radial_step).sin() * radius;

        let vertex = Vertex { position: [x, y, 0.0], color };

        vertices.push(vertex);
    });

    vertices
}












