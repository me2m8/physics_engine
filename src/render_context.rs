use core::num;
use std::{collections::HashMap, default};

use cgmath::{Vector2, Vector3, Vector4};
use wgpu::{
    include_wgsl, vertex_attr_array, BufferAddress, ColorWrites, Device, FragmentState, FrontFace,
    PipelineCompilationOptions, PrimitiveState, PrimitiveTopology, Queue, ShaderModule, Surface,
    SurfaceConfiguration, VertexAttribute, VertexBufferLayout,
};

pub struct RenderContext {
    shaders: HashMap<String, ShaderModule>,
    pipelines: HashMap<PipelineType, wgpu::RenderPipeline>,
    rectangles: Vec<Instance>,
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
}

impl RenderContext {
    pub fn new(device: &Device, config: &SurfaceConfiguration) -> Self {
        let mut pipelines = HashMap::new();

        use super::include_many_wgsl;

        #[rustfmt::skip]
        let shader_descriptors = include_many_wgsl![
            "shaders/polygon_fill.wgsl"
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
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FilledPolygon pipeline"),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Front),
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
                buffers: &[FilledPolygonVertex::DESC],
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
        });

        pipelines.insert(PipelineType::PolygonFill, pipeline);

        Self {
            shaders,
            pipelines,
            rectangles: Default::default(),
        }
    }
}

#[macro_export]
/// Returns a vec of tuples mapping the shader path to the shader module descriptor
macro_rules! include_many_wgsl {
    [$($x: literal),*] => {
        [$((
            $x.to_string(),
            include_wgsl!($x)
        ))*]
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FilledPolygonVertex {
    position: [f32; 4],
    color: [f32; 4],
}

/// Describes a vertex to be sent to the gpu
pub trait Vertex {
    const ATTRIBS: &'static [VertexAttribute];
    const DESC: VertexBufferLayout<'static>;
}

impl Vertex for FilledPolygonVertex {
    const ATTRIBS: &'static [VertexAttribute] = &vertex_attr_array![0 => Float32x4, 1 => Float32x4];

    const DESC: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: Self::ATTRIBS,
    };
}

impl FilledPolygonVertex {
    fn new(position: impl Into<[f32; 3]>, color: impl Into<[f32; 4]>) -> Self {
        let p = position.into();
        Self {
            position: [p[0], p[1], p[2], 1.0],
            color: color.into(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineType {
    PolygonFill,
}
