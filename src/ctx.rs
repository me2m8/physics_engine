use wgpu::util::DeviceExt;

use crate::camera::Camera2D;
use crate::state::State;
use crate::instances::circle::*;

pub struct Ctx {
    pub circles: Vec<Circle>,
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub num_indicies: u32,
    pub num_instances: u32,
    pub camera: Camera2D,
}

fn random() -> f32 {
    rand::random::<f32>()
}

impl Ctx {
    pub fn new(state: &State) -> Self {

        let camera = Camera2D::new(state);

        let circles = (0..5).map(|_| Circle {
            position: [(random() - 0.5) * camera.width(), (random() - 0.5) * camera.height()],
            radius: random() * 100.0,
            color: [random(), random(), random(), 1.0],
        }).collect::<Vec<Circle>>();

        let shader_module = state.device().create_shader_module(wgpu::include_wgsl!("circle_shader.wgsl"));

        let vertex_buffer = state.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(CIRCLE_VERTS),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let index_buffer = state.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(CIRCLE_INDICIES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let instances = circles.iter().map(|e| e.to_raw()).collect::<Vec<RawCircle>>();

        let instance_buffer = state.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let num_instances = instances.len() as u32;

        let num_indicies = CIRCLE_INDICIES.len() as u32;

        let pipeline_layout = state.device().create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[&camera.bind_group_layout],
                push_constant_ranges: &[],
            }
        );

        let pipeline = state.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: state.sample_count(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[CircleVertex::desc(), RawCircle::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: state.config().format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::all(),
                })],
            }),
        });

        Self {
            circles,
            pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            num_indicies,
            num_instances,
            camera,
        }
    }

    pub fn update_instances(&mut self, state: &State) {
        let instances = self.circles.iter().map(|e| e.to_raw()).collect::<Vec<RawCircle>>();

        let instance_buffer = state.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let num_instances = instances.len() as u32;

        self.instance_buffer = instance_buffer;
        self.num_instances = num_instances;
    }

    pub fn physics_process(&mut self, state: &State) {
        self.circles.push(Circle {
            position: [(random() * 2.0 - 1.0) * state.config().width as f32, (random() * 2. - 1.) * state.config().height as f32],
            radius: random() * 100.0,
            color: [random(), random(), random(), 1.0],
        });

        self.update_instances(state);
    }
}
