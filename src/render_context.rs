use core::num;
use std::{collections::HashMap, default};

use cgmath::{Vector2, Vector4};
use wgpu::{
    include_wgsl, util::DeviceExt, vertex_attr_array, BufferAddress, BufferUsages, ColorWrites,
    Device, FragmentState, FrontFace, IndexFormat, PipelineCompilationOptions, PrimitiveState,
    PrimitiveTopology, Queue, RenderBundle, ShaderModule, Surface, SurfaceConfiguration,
    VertexAttribute, VertexBufferLayout,
};

pub struct RenderContext {
    device: wgpu::Device,
    queue: wgpu::Queue,

    shaders: HashMap<String, ShaderModule>,
    pipelines: HashMap<PipelineType, wgpu::RenderPipeline>,
    rectangles: Rectangles,
}

#[derive(Default, Clone, Debug)]
pub struct Rectangles {
    vertices: Vec<FilledPolygonVertex>,
    indicies: Vec<u16>,
}

pub struct Instance {
    transform: cgmath::Matrix4<f32>,
}

impl RenderContext {
    pub fn new(device: Device, queue: Queue, config: &SurfaceConfiguration) -> Self {
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

        pipelines.insert(
            PipelineType::PolygonFill,
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            }),
        );

        Self {
            device,
            queue,

            shaders,
            pipelines,
            rectangles: Rectangles::default(),
        }
    }

    pub fn draw_rectangle(
        &mut self,
        top_left: Vector2<f32>,
        bottom_right: Vector2<f32>,
        color: Vector4<f32>,
    ) {
        let tl = [top_left.x, top_left.y, 0.0];
        let tr = [bottom_right.x, top_left.y, 0.0];
        let bl = [top_left.x, bottom_right.y, 0.0];
        let br = [bottom_right.x, bottom_right.y, 0.0];

        let mut vertices = vec![
            FilledPolygonVertex::new(tr, color),
            FilledPolygonVertex::new(tl, color),
            FilledPolygonVertex::new(bl, color),
            FilledPolygonVertex::new(br, color),
        ];

        let offset = self.rectangles.vertices.len() as u16;
        let mut indicies = vec![
            offset,
            offset + 1,
            offset + 2,
            offset,
            offset + 2,
            offset + 3,
        ];

        self.rectangles.vertices.append(&mut vertices);
        self.rectangles.indicies.append(&mut indicies);
    }

    fn build_draw_commands(&mut self, config: &SurfaceConfiguration) -> Vec<RenderBundle> {
        let mut commands: Vec<RenderBundle> = vec![];

        // Render Rectangles
        if !self.rectangles.vertices.is_empty() {
            let mut encoder =
                self.device
                    .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                        label: Some("Bundle Encoder"),
                        color_formats: &[Some(config.format)],
                        depth_stencil: None,
                        sample_count: 1,
                        multiview: None,
                    });

            let vertex_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Rectangle Vertex Buffer"),
                    contents: bytemuck::cast_slice(&self.rectangles.vertices),
                    usage: BufferUsages::VERTEX,
                });

            let index_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Rectangle Index Buffer"),
                    contents: bytemuck::cast_slice(&self.rectangles.indicies),
                    usage: BufferUsages::INDEX,
                });

            let num_indicies = self.rectangles.indicies.len() as u32;

            dbg!(&self.rectangles.vertices);
            dbg!(&self.rectangles.indicies);

            encoder.set_pipeline(self.pipelines.get(&PipelineType::PolygonFill).unwrap());
            encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
            encoder.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint16);

            encoder.draw_indexed(0..num_indicies, 0, 0..1);

            commands.push(encoder.finish(&wgpu::RenderBundleDescriptor {
                label: Some("Rectangles Render Bundle"),
            }));

            self.rectangles.vertices.clear();
            self.rectangles.indicies.clear();
        }

        commands
    }

    pub fn draw(&mut self, surface: &Surface, config: &SurfaceConfiguration) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

        let output = surface.get_current_texture().unwrap();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        store: wgpu::StoreOp::Store,
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let commands = self.build_draw_commands(config);
            _render_pass.execute_bundles(commands.iter());
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
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
