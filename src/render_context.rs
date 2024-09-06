#![allow(unused)]
use core::num;
use std::{collections::HashMap, error::Error};

use cgmath::{vec2, Vector3, Vector4};
use wgpu::{
    include_wgsl, vertex_attr_array, BlendComponent, BlendState, Buffer, BufferAddress,
    BufferUsages, ColorWrites, Device, FragmentState, FrontFace, PipelineCompilationOptions,
    PrimitiveState, PrimitiveTopology, RenderPipeline, ShaderModule, SurfaceConfiguration,
    VertexAttribute, VertexBufferLayout,
};

use crate::{
    camera::{Camera, CameraState},
    MAX_VERTICES,
};

pub struct RenderContext<C>
where
    C: Camera + Sized,
{
    shaders: HashMap<String, ShaderModule>,
    pipelines: HashMap<PipelineType, wgpu::RenderPipeline>,

    camera: CameraState<C>,

    vertex_buffer: Buffer,
    index_buffer: Buffer,
}

impl<C> RenderContext<C>
where
    C: Camera + Sized,
{
    pub fn new(device: &Device, config: &SurfaceConfiguration) -> Self {
        let mut pipelines = HashMap::new();

        let viewport = vec2(config.width as f32, config.height as f32);
        let camera = CameraState::new(device, viewport);

        use super::include_many_wgsl;

        #[rustfmt::skip]
        let shader_descriptors = include_many_wgsl![
            "shaders/polygon_fill.wgsl",
            "shaders/circle_fill.wgsl",
            "shaders/circle_fade.wgsl",
            "shaders/zero_width_lines.wgsl"
        ];

        let shaders = shader_descriptors
            .into_iter()
            .map(|(file, desc)| {
                let module = device.create_shader_module(desc);
                (file, module)
            })
            .collect::<HashMap<String, ShaderModule>>();

        let general_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline_layout"),
            bind_group_layouts: &[camera.bind_group_layout()],
            push_constant_ranges: &[],
        });

        // PolygonFill pipeline
        pipelines.insert(
            PipelineType::PolygonFill,
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("FilledPolygon pipeline"),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                multiview: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                layout: Some(&general_layout),
                depth_stencil: None,
                cache: None,
                vertex: wgpu::VertexState {
                    module: shaders.get("shaders/polygon_fill.wgsl").unwrap(),
                    entry_point: "vs_main",
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Vertex::DESC],
                },
                fragment: Some(FragmentState {
                    module: shaders.get("shaders/polygon_fill.wgsl").unwrap(),
                    entry_point: "fs_main",
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
            }),
        );

        // CircleFill pipeline
        pipelines.insert(
            PipelineType::CircleFill,
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("FilledPolygon pipeline"),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                multiview: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                layout: Some(&general_layout),
                depth_stencil: None,
                cache: None,
                vertex: wgpu::VertexState {
                    module: shaders.get("shaders/circle_fill.wgsl").unwrap(),
                    entry_point: "vs_main",
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Vertex::DESC],
                },
                fragment: Some(FragmentState {
                    module: shaders.get("shaders/circle_fill.wgsl").unwrap(),
                    entry_point: "fs_main",
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
            }),
        );

        // CircleFade pipeline
        pipelines.insert(
            PipelineType::CircleFade,
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("FilledPolygon pipeline"),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                multiview: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                layout: Some(&general_layout),
                depth_stencil: None,
                cache: None,
                vertex: wgpu::VertexState {
                    module: shaders.get("shaders/circle_fade.wgsl").unwrap(),
                    entry_point: "vs_main",
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Vertex::DESC],
                },
                fragment: Some(FragmentState {
                    module: shaders.get("shaders/circle_fade.wgsl").unwrap(),
                    entry_point: "fs_main",
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
            }),
        );

        // ZeroWidthLines pipeline
        pipelines.insert(
            PipelineType::ZeroWidthLines,
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zero width line pipeline"),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Line,
                    conservative: false,
                },
                multiview: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                layout: Some(&general_layout),
                depth_stencil: None,
                cache: None,
                vertex: wgpu::VertexState {
                    module: shaders.get("shaders/zero_width_lines.wgsl").unwrap(),
                    entry_point: "vs_main",
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Vertex::DESC],
                },
                fragment: Some(FragmentState {
                    module: shaders.get("shaders/zero_width_lines.wgsl").unwrap(),
                    entry_point: "fs_main",
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
            }),
        );

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (size_of::<Vertex>() * crate::MAX_VERTICES) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (size_of::<u16>() * 2048) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            index_buffer,

            camera,

            shaders,
            pipelines,
        }
    }

    pub fn vertex_buffer(&self) -> &Buffer {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &Buffer {
        &self.index_buffer
    }

    pub fn pipeline(&self, pipeline_type: PipelineType) -> &RenderPipeline {
        self.pipelines.get(&pipeline_type).unwrap()
    }

    pub fn camera(&self) -> &CameraState<C> {
        &self.camera
    }
}

#[macro_export]
/// Returns an array of tuples mapping the shader path to the shader module descriptor
macro_rules! include_many_wgsl {
    [$($x: literal),*] => {
        [$((
            $x.to_string(),
            include_wgsl!($x)
        ),)*]
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub frag_coord: [f32; 2],
}

impl Vertex {
    fn new(
        position: impl Into<[f32; 3]>,
        color: impl Into<[f32; 4]>,
        frag_coord: impl Into<[f32; 2]>,
    ) -> Self {
        let p = position.into();
        Self {
            position: [p[0], p[1], p[2], 1.0],
            color: color.into(),
            frag_coord: frag_coord.into(),
        }
    }
    const ATTRIBS: &'static [VertexAttribute] =
        &vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x2];

    const DESC: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: Self::ATTRIBS,
    };
}

pub struct RawInstance {
    transform: cgmath::Matrix4<f32>,
    color: Vector4<f32>,
}

pub struct Instance {
    position: Vector3<f32>,
    pitch: f32,
    yaw: f32,
    roll: f32,
    x_scale: f32,
    y_scale: f32,
    color: Vector4<f32>,
}

impl Instance {
    const ATTRIBS: &'static [VertexAttribute] = &vertex_attr_array![
        0 => Float32x4, // Matrix4x4
        1 => Float32x4, // Matrix4x4
        2 => Float32x4, // Matrix4x4
        3 => Float32x4, // Matrix4x4
        4 => Float32x4, // Color
    ];

    const DESC: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<RawInstance>() as BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: Self::ATTRIBS,
    };
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineType {
    PolygonFill,
    CircleFill,
    CircleFade,
    ZeroWidthLines,
}

pub fn quad_indicies_from_verticies(vertices: &[Vertex]) -> Result<Vec<u16>, Box<dyn Error>> {
    let num_vertices = vertices.len();
    if num_vertices % 4 != 0 {
        return Err("Vertex amount not divisible by 4, so the vertices cannot form quads".into());
    }

    Ok((0..num_vertices as u16)
        .step_by(4)
        .fold(
            Vec::with_capacity(size_of::<u16>() * num_vertices * 3 / 2),
            |mut acc, i| {
                acc.push([i, i + 1, i + 2, i + 2, i + 3, i]);

                acc
            },
        )
        .concat())
}
