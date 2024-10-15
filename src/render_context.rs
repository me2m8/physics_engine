#![allow(unused)]
use core::num;
use std::{
    borrow::{BorrowMut, Cow},
    cell::RefCell,
    collections::HashMap,
    error::Error,
    fmt::Debug,
    num::NonZero,
};

use bytemuck::bytes_of;
use cgmath::{vec2, Vector2, Vector3, Vector4};
use itertools::Itertools;
use log::info;
use wgpu::{
    include_wgsl,
    util::{DeviceExt, RenderEncoder},
    vertex_attr_array, BlendComponent, BlendState, Buffer, BufferAddress, BufferSize, BufferUsages,
    ColorWrites, Device, FragmentState, FrontFace, PipelineCompilationOptions, PrimitiveState,
    PrimitiveTopology, Queue, QueueWriteBufferView, RenderPipeline, ShaderModule, Surface,
    SurfaceConfiguration, Texture, VertexAttribute, VertexBufferLayout, COPY_BUFFER_ALIGNMENT,
};

use crate::{
    camera::{Camera, Camera2D, CameraState},
    error::PrimitiveError,
    instance::{Instance, Instance2D},
    primitives::{create_arrow_template, create_circle_primitive, Primitive},
    MAX_CIRCLES, SAMPLE_COUNT,
};

use crate::shaders::{make_pipelines, PipelineType};

pub struct RenderContext<C>
where
    C: Camera + Sized,
{
    pipelines: HashMap<PipelineType, wgpu::RenderPipeline>,
    config: wgpu::SurfaceConfiguration,
    camera: CameraState<C>,

    buffers: BufferMaster,

    circles: Primitive<CircleVertex, Instance2D>,
    arrows: Primitive<ArrowVertex, Instance2D>,
}

impl<C> RenderContext<C>
where
    C: Camera + Sized,
{
    pub fn new(device: &Device, config: &SurfaceConfiguration) -> Self {
        let float_width = config.width as f32;
        let float_height = config.height as f32;

        // This scales the viewport such that the width becomes this amount in pixels
        let viewport_scale = crate::VIEWPORT_SCALE;

        let viewport = vec2(float_width, float_height) / float_width * viewport_scale;
        let camera = CameraState::new(device, viewport);

        let pipelines = make_pipelines(device, config, &camera);

        let circles = Primitive::new(create_circle_primitive(5.0));
        let arrows = Primitive::new(create_arrow_template(7.0, 2.0, 2.0, 2.0));

        let buffers = BufferMaster::new(device);

        Self {
            camera,
            pipelines,
            config: config.clone(),

            buffers,

            circles,
            arrows,
        }
    }

    pub fn begin_scene(&mut self, queue: &Queue) {
        self.circles().clear_instances();
        self.arrows().clear_instances();
    }

    pub fn present_scene(
        &mut self,
        queue: &Queue,
        device: &Device,
        surface: &Surface,
        msaa: &Texture,
    ) {
        // Creating command encoder for sending commands to the gpu
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // Getting the output texture and texture view
        let output = surface.get_current_texture().unwrap();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create view for multisampling
        let msaa_view = msaa.create_view(&wgpu::TextureViewDescriptor::default());

        // Write data to buffers
        self.buffers.reset();
        if self.arrows().has_instances() {
            self.buffers.add_primitives(queue, &self.arrows);
        }
        if self.circles().has_instances() {
            self.buffers.add_primitives(queue, &self.circles);
        }

        queue.submit([]);

        // Render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &msaa_view,
                    resolve_target: Some(&view),
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
            render_pass.set_vertex_buffer(0, self.buffers.vertex_buf.slice(..));
            render_pass.set_vertex_buffer(1, self.buffers.instance_buf.slice(..));
            render_pass
                .set_index_buffer(self.buffers.index_buf.slice(..), wgpu::IndexFormat::Uint16);

            for i in 0..self.buffers.primitive_count as usize {
                // Get offsets start and end
                let offsets = &self.buffers.offsets;
                let start = offsets[i];
                let end = offsets.get(i + 1).unwrap_or(&self.buffers.end_offsets);

                // Set the correct pipeline
                let pipeline = self.pipeline(self.buffers.pipelines[i]);
                render_pass.set_pipeline(pipeline);

                // Draw each primitive
                render_pass.draw_indexed(
                    start.index_offset..end.index_offset,
                    start.vertex_offset as i32,
                    start.instance_offset..end.instance_offset,
                );
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
        output.present();
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

    pub fn arrows(&self) -> &Primitive<ArrowVertex, Instance2D> {
        &self.arrows
    }
    pub fn circles(&self) -> &Primitive<CircleVertex, Instance2D> {
        &self.circles
    }
}

pub mod shapes {
    use std::borrow::BorrowMut;

    use cgmath::{vec3, vec4, Angle, Deg, Matrix2, Rad, Rotation2};

    use crate::camera::Camera2D;

    use super::*;

    /// Draws a 2D circle
    pub fn draw_circle_2d(
        render_context: &RenderContext<Camera2D>,
        p: Vector2<f32>,
        r: f32,
        color: Vector4<f32>,
    ) {
        let c = vec4(p.x, p.y, 0.0, 1.0);
        let rad = vec3(r, -r, 0.0);

        let bl = c - rad.xxzz();
        let br = c - rad.yxzz();
        let tr = c + rad.xxzz();
        let tl = c + rad.yxzz();

        #[rustfmt::skip]
        let vertices = [
            CircleVertex { p: bl.into(), c: color.into(), fc: [-1., -1.]},
            CircleVertex { p: br.into(), c: color.into(), fc: [-1.,  1.]},
            CircleVertex { p: tr.into(), c: color.into(), fc: [ 1.,  1.]},
            CircleVertex { p: tl.into(), c: color.into(), fc: [ 1., -1.]},
        ];
        let indicies = [0, 1, 2, 2, 3, 0];
    }
}

/// Keeps track of the buffers and their contents
pub struct BufferMaster {
    index_buf: Buffer,
    vertex_buf: Buffer,
    instance_buf: Buffer,

    pipelines: Vec<PipelineType>,
    offsets: Vec<BufferOffsets>,
    end_offsets: BufferOffsets,

    primitive_count: u64,
}

/// Organizes the offsets for the buffer master
#[derive(Clone, Copy, Debug, Default)]
struct BufferOffsets {
    index_offset: u32,
    vertex_offset: u32,
    instance_offset: u32,
}

impl BufferMaster {
    pub fn new(device: &Device) -> Self {
        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: crate::INDEX_BUF_SIZE,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: crate::VERT_BUF_SIZE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: crate::INST_BUF_SIZE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            index_buf,
            vertex_buf,
            instance_buf,

            pipelines: Default::default(),
            offsets: Default::default(),
            end_offsets: Default::default(),

            primitive_count: 0,
        }
    }

    pub fn add_primitives<V, I>(
        &mut self,
        queue: &Queue,
        primitives: &Primitive<V, I>,
    ) -> Result<(), Box<dyn Error>>
    where
        V: Vertex + Sized + bytemuck::Pod,
        I: Instance,
    {
        // Push the new offsets into the offsets vectors
        let offsets = self.end_offsets;

        let mut indx_view = queue
            .write_buffer_with(
                &self.index_buf,
                offsets.index_offset as u64,
                primitives.index_buffer_len(),
            )
            .ok_or(PrimitiveError::AppendIndx)?;
        let mut vert_view = queue
            .write_buffer_with(
                &self.vertex_buf,
                offsets.vertex_offset as u64,
                primitives.vertex_buffer_len(),
            )
            .ok_or(PrimitiveError::AppendVert)?;
        let mut inst_view = queue
            .write_buffer_with(
                &self.instance_buf,
                offsets.instance_offset as u64,
                primitives.instance_buffer_len(),
            )
            .ok_or(PrimitiveError::AppendInst)?;

        indx_view.copy_from_slice(primitives.index_bytes());
        vert_view.copy_from_slice(primitives.vertex_bytes());
        inst_view.copy_from_slice(&primitives.instance_bytes());

        self.pipelines.push(primitives.pipeline());
        self.offsets.push(offsets);

        self.end_offsets.index_offset = offsets.index_offset + primitives.num_indicies();
        self.end_offsets.vertex_offset = offsets.vertex_offset + primitives.num_vertices();
        self.end_offsets.instance_offset = offsets.instance_offset + primitives.num_instances();

        self.primitive_count += 1;

        Ok(())
    }

    /// Loses track of all data in the buffers.
    pub fn reset(&mut self) {
        self.end_offsets = Default::default();
        self.offsets.clear();
        self.primitive_count = 0;
    }
}

//
// VERTICES
//

pub trait Vertex: std::marker::Sized + Copy {
    const ATTRIBS: &'static [VertexAttribute];

    const DESC: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: Self::ATTRIBS,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CircleVertex {
    pub p: [f32; 4],
    pub c: [f32; 4],
    pub fc: [f32; 2],
}
impl Vertex for CircleVertex {
    const ATTRIBS: &'static [VertexAttribute] =
        &vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x2];
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub position: [f32; 4],
    pub color: [f32; 4],
}
impl Vertex for LineVertex {
    const ATTRIBS: &'static [VertexAttribute] = &vertex_attr_array![0 => Float32x4, 1 => Float32x4];
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ArrowVertex {
    pub p: [f32; 4],
    pub c: [f32; 4],
}
impl Vertex for ArrowVertex {
    const ATTRIBS: &'static [VertexAttribute] = &vertex_attr_array![0 => Float32x4, 1 => Float32x4];
}
