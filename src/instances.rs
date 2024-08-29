use crate::{camera::Camera2D, state::State};
use cgmath::{Vector2, Vector4};
use wgpu::util::DeviceExt;

pub mod circle {

    use super::*;

    #[derive(Copy, Clone, Debug)]
    pub struct Circle {
        pub position: Vector2<f32>,
        pub radius: f32,
        pub color: Vector4<f32>,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
    pub struct RawCircle {
        position: [f32; 2],
        radius: f32,
        color: [f32; 4],
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
    pub struct CircleVertex {
        frag_coord: [f32; 2],
    }

    pub struct CircleRender {
        pub pipeline: wgpu::RenderPipeline,
        pub vertex_buffer: wgpu::Buffer,
        pub index_buffer: wgpu::Buffer,
        pub instance_buffer: wgpu::Buffer,
        pub num_indicies: u32,
        pub num_instances: u32,
    }

    #[rustfmt::skip]
    pub const CIRCLE_VERTS: &[CircleVertex] = &[
        CircleVertex { frag_coord: [ 1.0,  1.0] },
        CircleVertex { frag_coord: [-1.0,  1.0] },
        CircleVertex { frag_coord: [-1.0, -1.0] },
        CircleVertex { frag_coord: [ 1.0, -1.0] },
    ];

    #[rustfmt::skip]
    pub const CIRCLE_INDICIES: &[u16] = &[
        0, 1, 2, 
        0, 2, 3
    ];

    impl CircleVertex {
        const ATTRIBS: &'static [wgpu::VertexAttribute] = &wgpu::vertex_attr_array![0 => Float32x2];

        pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CircleVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: Self::ATTRIBS,
            }
        }
    }

    impl Circle {
        pub fn new(position: Vector2<f32>, radius: f32, color: Vector4<f32>) -> Self {
            Self {
                position,
                radius,
                color,
            }
        }

        pub fn to_raw(&self) -> RawCircle {
            RawCircle {
                position: self.position.into(),
                radius: self.radius,
                color: self.color.into(),
            }
        }
    }

    impl RawCircle {
        const ATTRIBS: &'static [wgpu::VertexAttribute] =
            &wgpu::vertex_attr_array![1 => Float32x2, 2 => Float32, 3 => Float32x4];

        pub fn desc() -> wgpu::VertexBufferLayout<'static> {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<RawCircle>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: Self::ATTRIBS,
            }
        }
    }

    impl CircleRender {
        pub fn new(state: &State, camera: &Camera2D) -> Self {
            let shader_module = state
                .device()
                .create_shader_module(wgpu::include_wgsl!("circle.wgsl"));

            let vertex_buffer =
                state
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Vertex Buffer"),
                        contents: bytemuck::cast_slice(CIRCLE_VERTS),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            let index_buffer =
                state
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Index Buffer"),
                        contents: bytemuck::cast_slice(CIRCLE_INDICIES),
                        usage: wgpu::BufferUsages::INDEX,
                    });

            let instances: Vec<RawCircle> = vec![];

            let instance_buffer =
                state
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer"),
                        contents: bytemuck::cast_slice(&instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            let num_instances = 1;

            let num_indicies = CIRCLE_INDICIES.len() as u32;

            let pipeline_layout =
                state
                    .device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Pipeline Layout"),
                        bind_group_layouts: &[&camera.bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline = state
                .device()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                vertex_buffer,
                index_buffer,
                instance_buffer,
                num_indicies,
                num_instances,
                pipeline,
            }
        }

        pub fn update_instances(&mut self, state: &State, circles: &[Circle]) {
            let instances = circles
                .iter()
                .map(|e| e.to_raw())
                .collect::<Vec<RawCircle>>();

            let instance_buffer =
                state
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer"),
                        contents: bytemuck::cast_slice(&instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            let num_instances = instances.len() as u32;

            self.instance_buffer = instance_buffer;
            self.num_instances = num_instances;
        }
    }
}

pub mod wireframe {
    use cgmath::{InnerSpace, Vector2, Zero};
    use wgpu::{util::DeviceExt, vertex_attr_array};

    use crate::{camera::Camera2D, state::State};

    #[derive(Clone, Debug)]
    pub struct Wireframe {
        vertices: Vec<Vector2<f32>>,
        line_width: f32,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
    pub struct WireframeVertex {
        pub position: [f32; 2],
        pub color: [f32; 4],
    }

    pub struct WireframeRender {
        pub pipeline: wgpu::RenderPipeline,
        pub vertex_buffer: wgpu::Buffer,
        pub index_buffer: wgpu::Buffer,
        pub instance_buffer: wgpu::Buffer,
        pub num_indicies: u32,
        pub num_instances: u32,
    }

    impl WireframeVertex {
        const ATTRIBS: &'static [wgpu::VertexAttribute] =
            &vertex_attr_array![0 => Float32x2, 1 => Float32x4];

        pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<WireframeVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: Self::ATTRIBS,
            }
        }
    }

    impl Wireframe {
        pub fn new(vertices: impl Into<Vec<Vector2<f32>>>, line_width: f32) -> Self {
            Self {
                vertices: vertices.into(),
                line_width,
            }
        }

        pub fn set_vertices(&mut self, vertices: impl Into<Vec<Vector2<f32>>>) {
            self.vertices = vertices.into();
        }

        fn vertices(&self) -> &[Vector2<f32>] {
            &self.vertices
        }

        pub fn to_vertices(&self) -> (Vec<WireframeVertex>, Vec<u16>) {
            use std::f32::consts::FRAC_1_SQRT_2;

            const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

            let vertices = self.vertices();
            let num_vertices = vertices.len() as isize;

            let diagonal_width = self.line_width * FRAC_1_SQRT_2;

            let pos_mod = |a: isize, b: isize| (((a % b) + b) % b) as usize;

            let calculate_offset_normal = |i: isize| -> Vector2<f32> {
                let rel_next = vertices[pos_mod(i + 1, num_vertices)] - vertices[i as usize];
                let rel_prior = vertices[pos_mod(i - 1, num_vertices)] - vertices[i as usize];

                let mag = rel_next.magnitude().min(rel_prior.magnitude());

                let diff = rel_prior.normalize() * mag - rel_next.normalize() * mag;
                Vector2::new(-diff.y, diff.x).normalize()
            };

            let mut new_vertices: Vec<WireframeVertex> = Vec::new();

            (0..num_vertices).for_each(|i: isize| {
                let perp_normal = calculate_offset_normal(i);

                new_vertices.push(WireframeVertex {
                    position: (vertices[i as usize] + perp_normal * diagonal_width).into(),
                    color: WHITE,
                });
                new_vertices.push(WireframeVertex {
                    position: (vertices[i as usize] - perp_normal * diagonal_width).into(),
                    color: WHITE,
                });
            });

            let mut indicies: Vec<u16> = Vec::new();

            for i in 0..num_vertices {
                let i = (i * 2) as u16;

                // Funny
                indicies.push(i);
                indicies.push(pos_mod(i as isize + 2, num_vertices * 2) as u16);
                indicies.push(pos_mod(i as isize + 1, num_vertices * 2) as u16);
                indicies.push(pos_mod(i as isize + 2, num_vertices * 2) as u16);
                indicies.push(pos_mod(i as isize + 3, num_vertices * 2) as u16);
                indicies.push(pos_mod(i as isize + 1, num_vertices * 2) as u16);
            }

            (new_vertices, indicies)

            // let new_vertices = vec![
            //     WireframeVertex { position: (vertices[i] + perp_normal * diagonal_width).into(), color: WHITE },
            //     WireframeVertex { position: (vertices[i + 1 % num_vertices] + next_perp_normal * diagonal_width).into(), color: WHITE },
            //     WireframeVertex { position: (vertices[i + 1 % num_vertices] - next_perp_normal * diagonal_width).into(), color: WHITE },
            //     WireframeVertex { position: (vertices[i] - perp_normal * diagonal_width).into(), color: WHITE },
            // ];
        }
    }

    impl WireframeRender {
        pub fn new(state: &State, camera: &Camera2D) -> Self {
            let shader_module = state
                .device()
                .create_shader_module(wgpu::include_wgsl!("wireframe.wgsl"));

            let vertex_buffer =
                state
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Vertex Buffer"),
                        contents: &[],
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            let index_buffer =
                state
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Index Buffer"),
                        contents: &[],
                        usage: wgpu::BufferUsages::INDEX,
                    });

            let instance_buffer =
                state
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer"),
                        contents: &[],
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            let num_instances = 1;
            let num_indicies = 0;

            let pipeline_layout =
                state
                    .device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Pipeline Layout"),
                        bind_group_layouts: &[&camera.bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline = state
                .device()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        buffers: &[WireframeVertex::desc()],
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
                vertex_buffer,
                index_buffer,
                instance_buffer,
                num_indicies,
                num_instances,
                pipeline,
            }
        }

        pub fn update_buffers(&mut self, state: &State, wireframes: &[Wireframe]) {
            let (vertices, indicies): (Vec<WireframeVertex>, Vec<u16>) =
                wireframes.iter().map(|e| e.to_vertices()).enumerate().fold(
                    (vec![], vec![]),
                    |mut acc, (i, (mut verticies, indicies))| {
                        acc.0.append(&mut verticies);
                        acc.1
                            .append(&mut indicies.iter().map(|e| e + i as u16).collect());

                        acc
                    },
                );

            let vertex_buffer =
                state
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Wireframe Vertex Buffer"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            let index_buffer =
                state
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Wireframe Index Buffer"),
                        contents: bytemuck::cast_slice(&indicies),
                        usage: wgpu::BufferUsages::INDEX,
                    });

            let num_indicies = indicies.len() as u32;

            self.vertex_buffer = vertex_buffer;
            self.index_buffer = index_buffer;
            self.num_indicies = num_indicies;
        }
    }
}
