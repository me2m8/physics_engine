use std::collections::HashMap;
use wgpu::{
    include_wgsl, ColorWrites, FragmentState, FrontFace, PipelineCompilationOptions,
    PrimitiveState, PrimitiveTopology, SurfaceConfiguration,
};

use crate::{
    camera::{Camera, CameraState}, instance::{Instance, Instance2D, RawInstance2D}, render_context::{ArrowVertex, CircleVertex, LineVertex, Vertex}, SAMPLE_COUNT
};

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

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineType {
    Arrow2D,
    CircleFill,
    CircleFade,
    ZeroWidthLines,
}

pub fn make_pipelines<T: Camera + Sized>(
    device: &wgpu::Device,
    config: &SurfaceConfiguration,
    camera: &CameraState<T>,
) -> HashMap<PipelineType, wgpu::RenderPipeline> {
    use super::include_many_wgsl;

    let shader_descriptors = include_many_wgsl![
        "arrow2d.wgsl",
        "circle_fill.wgsl",
        "circle_fade.wgsl",
        "zero_width_lines.wgsl"
    ];

    let mut pipelines = HashMap::new();

    let shaders = shader_descriptors
        .into_iter()
        .map(|(file, desc)| {
            let module = device.create_shader_module(desc);
            (file, module)
        })
        .collect::<HashMap<String, wgpu::ShaderModule>>();

    let general_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render pipeline_layout"),
        bind_group_layouts: &[camera.bind_group_layout()],
        push_constant_ranges: &[],
    });

    // Arrow2D pipeline
    pipelines.insert(
        PipelineType::Arrow2D,
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Arrow2d pipeline"),
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
                count: SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            layout: Some(&general_layout),
            depth_stencil: None,
            cache: None,
            vertex: wgpu::VertexState {
                module: shaders.get("arrow2d.wgsl").unwrap(),
                entry_point: "vs_main",
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[ArrowVertex::DESC, Instance2D::DESC],
            },
            fragment: Some(FragmentState {
                module: shaders.get("arrow2d.wgsl").unwrap(),
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
                count: SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            layout: Some(&general_layout),
            depth_stencil: None,
            cache: None,
            vertex: wgpu::VertexState {
                module: shaders.get("circle_fill.wgsl").unwrap(),
                entry_point: "vs_main",
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[CircleVertex::DESC],
            },
            fragment: Some(FragmentState {
                module: shaders.get("circle_fill.wgsl").unwrap(),
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
                count: SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            layout: Some(&general_layout),
            depth_stencil: None,
            cache: None,
            vertex: wgpu::VertexState {
                module: shaders.get("circle_fade.wgsl").unwrap(),
                entry_point: "vs_main",
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[CircleVertex::DESC],
            },
            fragment: Some(FragmentState {
                module: shaders.get("circle_fade.wgsl").unwrap(),
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
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            multiview: None,
            multisample: wgpu::MultisampleState {
                count: SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            layout: Some(&general_layout),
            depth_stencil: None,
            cache: None,
            vertex: wgpu::VertexState {
                module: shaders.get("zero_width_lines.wgsl").unwrap(),
                entry_point: "vs_main",
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[LineVertex::DESC],
            },
            fragment: Some(FragmentState {
                module: shaders.get("zero_width_lines.wgsl").unwrap(),
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

    pipelines
}
