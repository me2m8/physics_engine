use std::collections::HashMap;

use cgmath::{Vector2, Vector3, Vector4};
use wgpu::{
    include_wgsl, vertex_attr_array, BufferAddress, FragmentState, FrontFace,
    PipelineCompilationOptions, PrimitiveState, PrimitiveTopology, RenderPipeline, ShaderModule,
    VertexAttribute, VertexBufferLayout,
};

use crate::application::WindowState;

pub struct RenderContext {
    shaders: HashMap<String, ShaderModule>,
    pipelines: HashMap<PipelineType, wgpu::RenderPipeline>,
    draw_calls: Vec<DrawCall>,
}

pub struct DrawCall {
    pipeline: PipelineType,
    vertices: Vec<FilledPolygonVertex>,
    indicies: Vec<u16>,
    instances: Vec<Instance>,
}

pub struct Instance {
    transform: cgmath::Matrix4<f32>,
}

impl RenderContext {
    pub fn new(state: &WindowState) -> Self {
        let mut pipelines = HashMap::new();
        use super::include_many_wgsl;
        let shaders = include_many_wgsl!["shaders/polygon_fill.wgsl"]
            .into_iter()
            .map(|(file, desc)| {
                let module = state.device().create_shader_module(desc);
                (file, module)
            })
            .collect::<HashMap<String, ShaderModule>>();

        let general_layout =
            state
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render pipeline_layout"),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                });

        pipelines.insert(
            PipelineType::PolygonFill,
            state
                .device()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        buffers: &[FilledPolygonVertex::DESC],
                    },
                    fragment: Some(FragmentState {
                        module: shaders.get("shaders/polygon_fill.wgsl").unwrap(),
                        entry_point: "fs_main",
                        compilation_options: PipelineCompilationOptions::default(),
                        targets: &[],
                    }),
                }),
        );

        Self {
            shaders,
            pipelines,
            draw_calls: Default::default(),
        }
    }

    pub fn draw_rectangle_2d(
        &mut self,
        top_left: Vector2<f32>,
        bottom_right: Vector2<f32>,
        color: Vector4<f32>,
    ) {
        let tl = [top_left.x, top_left.y, 0.0];
        let tr = [bottom_right.x, top_left.y, 0.0];
        let bl = [top_left.x, bottom_right.y, 0.0];
        let br = [bottom_right.x, bottom_right.y, 0.0];

        let vertices = vec![
            FilledPolygonVertex::new(tr, color),
            FilledPolygonVertex::new(tl, color),
            FilledPolygonVertex::new(bl, color),
            FilledPolygonVertex::new(br, color),
        ];

        let indicies = vec![0, 1, 2, 0, 2, 3];

        let instances = vec![];

        self.draw_calls.push(DrawCall {
            pipeline: PipelineType::PolygonFill,
            vertices,
            indicies,
            instances,
        });
    }
}

#[macro_export]
macro_rules! include_many_wgsl {
    [$($x: literal),*] => {
        [$((
            $x.to_string(),
            include_wgsl!($x)
        ))*]
    };
}

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
