#![allow(unused)]
use core::num;
use std::{borrow::Cow, collections::HashMap, error::Error, num::NonZero};

use cgmath::{vec2, Vector2, Vector3, Vector4};
use itertools::Itertools;
use wgpu::{
    include_wgsl, util::RenderEncoder, vertex_attr_array, BlendComponent, BlendState, Buffer,
    BufferAddress, BufferUsages, ColorWrites, Device, FragmentState, FrontFace,
    PipelineCompilationOptions, PrimitiveState, PrimitiveTopology, Queue, QueueWriteBufferView,
    RenderPipeline, ShaderModule, Surface, SurfaceConfiguration, VertexAttribute,
    VertexBufferLayout,
};

use crate::{
    camera::{Camera, CameraState},
    MAX_QUADS,
};

use crate::shaders::{make_pipelines, PipelineType};

pub struct RenderContext<C>
where
    C: Camera + Sized,
{
    pipelines: HashMap<PipelineType, wgpu::RenderPipeline>,

    camera: CameraState<C>,

    // This is safe because the buffer view is dropped before the buffer
    circle_vb_view: Option<QueueWriteBufferView<'static>>,
    circle_ib_view: Option<QueueWriteBufferView<'static>>,

    line_vb_view: Option<QueueWriteBufferView<'static>>,
    line_ib_view: Option<QueueWriteBufferView<'static>>,

    pub circle_vertices: Vec<QuadVertex>,
    circle_vb: Buffer,
    circle_ib: Buffer,

    pub line_vertices: Vec<LineVertex>,
    line_vb: Buffer,
    line_ib: Buffer,

    pub arrow_vertices: Vec<LineVertex>,
    arrow_vb: Buffer,
    arrow_ib: Buffer,
}

impl<C> RenderContext<C>
where
    C: Camera + Sized,
{
    pub fn new(device: &Device, config: &SurfaceConfiguration) -> Self {
        let float_width = config.width as f32;
        let float_height = config.height as f32;

        // This scales the viewport such that the width becomes this amount in pixels
        let viewport_scale = 200.0;

        let viewport = vec2(float_width, float_height) / float_width * viewport_scale;
        let camera = CameraState::new(device, viewport);

        let pipelines = make_pipelines(device, config, &camera);

        let circle_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (size_of::<[QuadVertex; 4]>() * crate::MAX_QUADS) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let circle_ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (size_of::<u16>() * crate::MAX_QUADS * 6) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let line_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (size_of::<[LineVertex; 2]>() * crate::MAX_LINES) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let line_ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (size_of::<u16>() * crate::MAX_LINES * 2) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        /// TODO: Fix arrow buffers
        let arrow_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (size_of::<[QuadVertex; 4]>() * crate::MAX_QUADS) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let arrow_ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (size_of::<u16>() * crate::MAX_QUADS * 6) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            circle_vertices: Default::default(),
            circle_vb,
            circle_ib,

            line_vertices: Default::default(),
            line_vb,
            line_ib,

            circle_vb_view: None,
            circle_ib_view: None,
            line_vb_view: None,
            line_ib_view: None,

            camera,

            pipelines,
        }
    }

    /// This function currently does nothing and just panics
    pub fn begin_scene(&mut self, queue: &Queue) {
        panic!("Currently does nothing")
    }

    fn init_circle_buffer_views(&mut self, queue: &Queue, num_vertices: usize) {
        // NOTE: This is safe because the views are dropped before the buffers
        self.circle_vb_view = Some(unsafe {
            std::mem::transmute::<wgpu::QueueWriteBufferView<'_>, wgpu::QueueWriteBufferView<'_>>(
                queue
                    .write_buffer_with(
                        self.circle_vb(),
                        0,
                        NonZero::new((size_of::<QuadVertex>() * num_vertices) as u64).unwrap(),
                    )
                    .unwrap(),
            )
        });
        self.circle_ib_view = Some(unsafe {
            std::mem::transmute::<wgpu::QueueWriteBufferView<'_>, wgpu::QueueWriteBufferView<'_>>(
                queue
                    .write_buffer_with(
                        self.circle_ib(),
                        0,
                        NonZero::new((size_of::<u16>() * num_vertices * 3 / 2) as u64).unwrap(),
                    )
                    .unwrap(),
            )
        });
    }
    fn init_line_buffer_views(&mut self, queue: &Queue, num_vertices: usize) {
        // NOTE: This is safe because the views are dropped before the buffers
        self.line_vb_view = Some(unsafe {
            std::mem::transmute::<wgpu::QueueWriteBufferView<'_>, wgpu::QueueWriteBufferView<'_>>(
                queue
                    .write_buffer_with(
                        self.line_vb(),
                        0,
                        NonZero::new((size_of::<LineVertex>() * num_vertices) as u64).unwrap(),
                    )
                    .unwrap(),
            )
        });
        self.line_ib_view = Some(unsafe {
            std::mem::transmute::<wgpu::QueueWriteBufferView<'_>, wgpu::QueueWriteBufferView<'_>>(
                queue
                    .write_buffer_with(
                        self.line_ib(),
                        0,
                        NonZero::new((size_of::<u16>() * num_vertices * 2) as u64).unwrap(),
                    )
                    .unwrap(),
            )
        });
    }

    pub fn present_scene(&mut self, queue: &Queue, device: &Device, surface: &Surface) {
        // writing the buffers with stored vertices
        let mut circle_indicies = 0;
        if !self.circle_vertices.is_empty() {
            circle_indicies = self.write_circle_buffer(queue);
        }
        let mut line_indicies = 0;
        if !self.line_vertices.is_empty() {
            line_indicies = self.write_line_buffer(queue);
        }

        self.write_camera_buffer(queue);

        queue.submit([]);

        // Creating command encoder for sending commands to the gpu
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // Getting the output texture and texture view
        let output = surface.get_current_texture().unwrap();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        store: wgpu::StoreOp::Store,
                        load: wgpu::LoadOp::Clear(crate::CLEAR_COLOR),
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Set the camera bind group
            render_pass.set_bind_group(0, self.camera.bind_group(), &[]);

            // Draw the Lines
            render_pass.set_pipeline(self.pipeline(PipelineType::ZeroWidthLines));

            render_pass.set_vertex_buffer(0, self.line_vb.slice(..));
            render_pass.set_index_buffer(self.line_ib.slice(..), wgpu::IndexFormat::Uint16);

            render_pass.draw_indexed(0..line_indicies, 0, 0..1);

            // Draw the Circles
            render_pass.set_pipeline(self.pipeline(PipelineType::CircleFill));

            render_pass.set_vertex_buffer(0, self.circle_vb.slice(..));
            render_pass.set_index_buffer(self.circle_ib.slice(..), wgpu::IndexFormat::Uint16);

            render_pass.draw_indexed(0..circle_indicies, 0, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    /// Writes the circle vertex buffer and index buffer and returns the number of indicies
    fn write_circle_buffer(&mut self, queue: &Queue) -> u32 {
        self.init_circle_buffer_views(queue, self.circle_vertices.len());

        let indicies = circle_indicies_from_verticies(&self.circle_vertices).unwrap();

        self.circle_vb_view
            .as_mut()
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(&self.circle_vertices));
        self.circle_ib_view
            .as_mut()
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(&indicies));

        self.circle_vb_view = None;
        self.circle_ib_view = None;

        self.circle_vertices.clear();

        indicies.len() as u32
    }

    /// Writes the line vertex buffer and index buffer and returns the number of indicies
    fn write_line_buffer(&mut self, queue: &Queue) -> u32 {
        self.init_line_buffer_views(queue, self.line_vertices.len());

        let indicies = line_indicies_from_vertices(&self.line_vertices, true).unwrap();

        self.line_vb_view
            .as_mut()
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(&self.line_vertices));
        self.line_ib_view
            .as_mut()
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(&indicies));

        self.line_vb_view = None;
        self.line_ib_view = None;

        self.line_vertices.clear();

        indicies.len() as u32
    }

    /// Writes the camera uniform buffer
    fn write_camera_buffer(&mut self, queue: &Queue) {
        let mut buffer_view = queue
            .write_buffer_with(
                self.camera.uniform_buffer(),
                0,
                NonZero::new(size_of::<C::Raw>() as u64).unwrap(),
            )
            .unwrap();

        buffer_view
            .as_mut()
            .copy_from_slice(bytemuck::cast_slice(&[self.camera().to_raw()]));
    }

    pub fn pipeline(&self, pipeline_type: PipelineType) -> &RenderPipeline {
        self.pipelines.get(&pipeline_type).unwrap()
    }

    pub fn camera(&self) -> &CameraState<C> {
        &self.camera
    }

    /// Gets the size of the camera viewport
    pub fn viewport_size(&self) -> Vector2<f32> {
        self.camera.viewport_size()
    }

    /// Gets a reference to the circle vertex buffer
    pub fn circle_vb(&self) -> &Buffer {
        &self.circle_vb
    }

    /// Gets a reference to the circle index buffer
    pub fn circle_ib(&self) -> &Buffer {
        &self.circle_ib
    }

    /// Gets a reference to the line vertex buffer
    pub fn line_vb(&self) -> &Buffer {
        &self.line_vb
    }

    /// Gets a reference to the line index buffer
    pub fn line_ib(&self) -> &Buffer {
        &self.line_ib
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuadVertex {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub frag_coord: [f32; 2],
}

impl QuadVertex {
    pub fn new(
        position: impl Into<[f32; 2]>,
        color: impl Into<[f32; 4]>,
        frag_coord: impl Into<[f32; 2]>,
    ) -> Self {
        let p = position.into();
        Self {
            position: [p[0], p[1], 0.0, 1.0],
            color: color.into(),
            frag_coord: frag_coord.into(),
        }
    }
    const ATTRIBS: &'static [VertexAttribute] =
        &vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x2];

    pub const DESC: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: Self::ATTRIBS,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub position: [f32; 4],
    pub color: [f32; 4],
}

impl LineVertex {
    pub fn new(position: impl Into<[f32; 2]>, color: impl Into<[f32; 4]>) -> Self {
        let p = position.into();
        Self {
            position: [p[0], p[1], 0.0, 1.0],
            color: color.into(),
        }
    }
    const ATTRIBS: &'static [VertexAttribute] = &vertex_attr_array![0 => Float32x4, 1 => Float32x4];

    pub const DESC: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: Self::ATTRIBS,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ArrowVertex {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub frag_coord: [f32; 2],
}

impl ArrowVertex {
    pub fn new(
        position: impl Into<[f32; 2]>,
        color: impl Into<[f32; 4]>,
        frag_coord: impl Into<[f32; 2]>,
    ) -> Self {
        let p = position.into();
        Self {
            position: [p[0], p[1], 0.0, 1.0],
            color: color.into(),
            frag_coord: frag_coord.into(),
        }
    }
    const ATTRIBS: &'static [VertexAttribute] =
        &vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x2];

    pub const DESC: VertexBufferLayout<'static> = VertexBufferLayout {
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

pub fn circle_indicies_from_verticies(vertices: &[QuadVertex]) -> Result<Vec<u16>, Box<dyn Error>> {
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

/// Creates indicies for a line single connected line from vertices
/// if loop_back is set to true, it will connect the first with the last vertex
pub fn line_indicies_from_vertices(
    vertices: &[LineVertex],
    loop_back: bool,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let num_vertices = vertices.len();
    if num_vertices == 1 {
        return Err("Cannot form lines from a single vertex".into());
    }

    Ok((0..(num_vertices - (!loop_back as usize)) as u16)
        .map(|i| [i, (i + 1) % num_vertices as u16])
        .collect_vec()
        .concat())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_indicies_from_vertices_loopback() {
        let vertices = vec![
            LineVertex {
                position: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            LineVertex {
                position: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            LineVertex {
                position: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            LineVertex {
                position: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ];

        let indicies = line_indicies_from_vertices(&vertices, true).unwrap();

        assert_eq!(indicies, vec![0, 1, 1, 2, 2, 3, 3, 0,])
    }

    #[test]
    fn test_line_indicies_from_vertices_no_loopback() {
        let vertices = vec![
            LineVertex {
                position: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            LineVertex {
                position: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            LineVertex {
                position: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            LineVertex {
                position: [0.0, 0.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ];

        let indicies = line_indicies_from_vertices(&vertices, false).unwrap();

        assert_eq!(indicies, vec![0, 1, 1, 2, 2, 3])
    }
}
