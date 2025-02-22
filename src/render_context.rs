#![allow(unused)]
use core::num;
use std::{
    borrow::Cow, cell::RefCell, collections::HashMap, error::Error, fmt::Debug, num::NonZero,
};

use bytemuck::bytes_of;
use cgmath::{vec2, Vector2, Vector3, Vector4};
use image::Primitive;
use itertools::Itertools;
use log::info;
use wgpu::{
    include_wgsl,
    util::{DeviceExt, RenderEncoder},
    vertex_attr_array, BlendComponent, BlendState, Buffer, BufferAddress, BufferSize, BufferUsages,
    ColorWrites, Device, FragmentState, FrontFace, PipelineCompilationOptions, PrimitiveState,
    PrimitiveTopology, Queue, QueueWriteBufferView, RenderPass, RenderPipeline, ShaderModule,
    Surface, SurfaceConfiguration, Texture, VertexAttribute, VertexBufferLayout,
    COPY_BUFFER_ALIGNMENT,
};

use crate::{
    camera::{Camera, Camera2D, CameraState},
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

    circles: RefCell<DrawPrimitive<CircleVertex>>,
    thin_lines: RefCell<DrawPrimitive<LineVertex>>,
    arrows: RefCell<DrawPrimitive<ArrowVertex>>,
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

        let circles = RefCell::new(DrawPrimitive::new(device, 4, 6, crate::MAX_CIRCLES));
        let thin_lines = RefCell::new(DrawPrimitive::new(device, 2, 2, crate::MAX_LINES));
        let arrows = RefCell::new(DrawPrimitive::new(device, 7, 9, crate::MAX_ARROWS));

        Self {
            camera,
            pipelines,
            config: config.clone(),

            circles,
            thin_lines,
            arrows,
        }
    }

    /// This function currently does nothing and just panics
    pub fn begin_scene(&mut self, queue: &Queue) {
        panic!("Currently does nothing")
    }

    pub fn present_scene(
        &mut self,
        queue: &Queue,
        device: &Device,
        surface: &Surface,
        msaa: &Texture,
    ) {
        // Populate buffers if primitive has been drawn
        self.circles.get_mut().populate_buffers(queue);
        self.thin_lines.get_mut().populate_buffers(queue);
        self.arrows.get_mut().populate_buffers(queue);

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

        let msaa_view = msaa.create_view(&wgpu::TextureViewDescriptor::default());

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

            self.circles
                .borrow_mut()
                .render_primitive(&mut render_pass, self.pipeline(PipelineType::CircleFill));
            self.thin_lines.borrow_mut().render_primitive(
                &mut render_pass,
                self.pipeline(PipelineType::ZeroWidthLines),
            );
            self.arrows
                .borrow_mut()
                .render_primitive(&mut render_pass, self.pipeline(PipelineType::PolygonFill));
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
}

//
// PRIMITIVES
//

pub struct DrawPrimitive<V: Vertex + Sized + bytemuck::Pod> {
    vertices: Vec<V>,
    indices: Vec<u16>,
    ib: Buffer,
    vb: Buffer,

    num_indices: u32,
    in_use: bool,
}

impl<T> DrawPrimitive<T>
where
    T: Vertex + Sized + bytemuck::Pod + Debug,
{
    fn new(device: &Device, num_verts: usize, num_ind: usize, max_prim: usize) -> Self {
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (size_of::<T>() * max_prim * num_verts) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (size_of::<u16>() * max_prim * num_ind) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertices: Default::default(),
            indices: Default::default(),

            vb,
            ib,

            num_indices: 0,
            in_use: false,
        }
    }

    /// Appends the vertices and indices of a primitive to the vertex and index vectors
    pub fn add_primitive(&mut self, vertices: &[T], indices: &[u16]) {
        self.add_indices(indices);
        self.add_vertices(vertices);
    }
    fn add_vertices(&mut self, vertices: &[T]) {
        self.vertices.extend(vertices.iter());
    }
    fn add_indices(&mut self, indices: &[u16]) {
        let num_vertices = self.vertices.len() as u16;
        self.indices
            .extend(indices.iter().map(|i| i + num_vertices));
    }

    /// Populates the buffers with data. If there is no data, this function just returns
    pub fn populate_buffers(&mut self, queue: &Queue) {
        if !self.has_data() {
            return;
        }

        let num_vertices = self.vertices.len();
        let num_indices = self.indices.len();

        let vb_bytes_size = (num_vertices * size_of::<T>()) as u64;
        let vb_bytes_parity = vb_bytes_size % COPY_BUFFER_ALIGNMENT;

        let ib_bytes_size = (num_indices * size_of::<u16>()) as u64;
        let ib_bytes_parity = ib_bytes_size % COPY_BUFFER_ALIGNMENT;

        let mut vb_bytes = bytemuck::cast_slice::<T, u8>(&self.vertices).to_vec();
        let mut ib_bytes = bytemuck::cast_slice::<u16, u8>(&self.indices).to_vec();

        (0..vb_bytes_parity).for_each(|_| vb_bytes.push(0));
        (0..ib_bytes_parity).for_each(|_| ib_bytes.push(0));

        let mut vb_view = queue.write_buffer_with(&self.vb, 0, unsafe {
            BufferSize::new_unchecked(vb_bytes_size + vb_bytes_parity)
        });
        let mut ib_view = queue.write_buffer_with(&self.ib, 0, unsafe {
            BufferSize::new_unchecked(ib_bytes_size + ib_bytes_parity)
        });

        vb_view.unwrap().as_mut().copy_from_slice(&vb_bytes);
        ib_view.unwrap().as_mut().copy_from_slice(&ib_bytes);

        self.num_indices = self.indices.len() as u32;

        self.vertices.clear();
        self.indices.clear();

        self.in_use = true;
    }

    /// Renders the primitive on the given render pass
    pub fn render_primitive(&mut self, render_pass: &mut RenderPass, pipeline: &RenderPipeline) {
        if !self.in_use {
            return;
        }

        info!("Drawing primitive");
        render_pass.set_pipeline(pipeline);

        render_pass.set_vertex_buffer(0, self.vb().slice(..));
        render_pass.set_index_buffer(self.ib().slice(..), wgpu::IndexFormat::Uint16);

        render_pass.draw_indexed(0..self.num_indices(), 0, 0..1);

        self.in_use = false;
    }

    pub fn ib(&self) -> &Buffer {
        &self.ib
    }
    pub fn vb(&self) -> &Buffer {
        &self.vb
    }

    /// Returns whether the given primitive has been drawn
    #[inline]
    pub fn is_in_use(&self) -> bool {
        self.in_use
    }

    /// Returns whether the given primitive has been drawn
    #[inline]
    pub fn has_data(&self) -> bool {
        !self.vertices.is_empty()
    }

    pub fn num_indices(&self) -> u32 {
        self.num_indices
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
        let indices = [0, 1, 2, 2, 3, 0];

        render_context
            .circles
            .borrow_mut()
            .add_primitive(&vertices, &indices);
    }

    /// Curries the [`draw_arrow_2d`] function to reduce the amount of arguments for repeditive use
    pub fn construct_arrow_2d(
        render_context: &RenderContext<Camera2D>,
        length: f32,
        line_thickness: f32,
        point_length: f32,
        point_width: f32,
    ) -> impl Fn(Vector2<f32>, Rad<f32>, Vector4<f32>) + '_ {
        move |p, dir, color| {
            draw_arrow_2d(
                render_context,
                p,
                dir,
                length,
                line_thickness,
                point_length,
                point_width,
                color,
            )
        }
    }

    /// Draws a 2D arrow
    #[allow(clippy::too_many_arguments)]
    pub fn draw_arrow_2d(
        render_context: &RenderContext<Camera2D>,
        p: Vector2<f32>,
        dir: Rad<f32>,
        length: f32,
        line_thickness: f32,
        point_length: f32,
        point_width: f32,
        color: Vector4<f32>,
    ) {
        let line_hw = line_thickness / 2.0;
        let line_length = length - point_length;
        let point_hw = point_width / 2.0;

        let line_bl = (rotate(vec2(0.0, -line_hw), dir) + p)
            .extend(0.0)
            .extend(1.0);
        let line_br = (rotate(vec2(line_length, -line_hw), dir) + p)
            .extend(0.0)
            .extend(1.0);
        let line_tr = (rotate(vec2(line_length, line_hw), dir) + p)
            .extend(0.0)
            .extend(1.0);
        let line_tl = (rotate(vec2(0.0, line_hw), dir) + p)
            .extend(0.0)
            .extend(1.0);

        let point_bottom = (rotate(vec2(line_length, -point_width), dir) + p)
            .extend(0.0)
            .extend(1.0);
        let point_tip = (rotate(vec2(length, 0.0), dir) + p).extend(0.0).extend(1.0);
        let point_top = (rotate(vec2(line_length, point_width), dir) + p)
            .extend(0.0)
            .extend(1.0);

        let c: [f32; 4] = color.into();

        #[rustfmt::skip]
        let vertices = [
            ArrowVertex { p: line_bl.into()     , c},
            ArrowVertex { p: line_br.into()     , c},
            ArrowVertex { p: line_tr.into()     , c},
            ArrowVertex { p: line_tl.into()     , c},
            ArrowVertex { p: point_bottom.into(), c},
            ArrowVertex { p: point_tip.into()   , c},
            ArrowVertex { p: point_top.into()   , c}
        ];

        let indices = [0, 1, 2, 2, 3, 0, 4, 5, 6];

        render_context
            .arrows
            .borrow_mut()
            .add_primitive(&vertices, &indices);
    }

    fn rotate(vec: Vector2<f32>, angle: Rad<f32>) -> Vector2<f32> {
        let (sin, cos) = angle.sin_cos();
        let rot_mat = Matrix2::new(cos, -sin, sin, cos);

        rot_mat * vec
    }

    pub fn draw_line_2d(
        render_context: &RenderContext<Camera2D>,
        start: Vector2<f32>,
        end: Vector2<f32>,
        color: Vector4<f32>,
    ) {
        let s = vec4(start.x, start.y, 0.0, 1.0);
        let e = vec4(end.x, end.y, 0.0, 1.0);

        #[rustfmt::skip]
        let vertices = [
            LineVertex { position: s.into(), color: color.into() },
            LineVertex { position: e.into(), color: color.into() },
        ];
        let indices = [0, 1];

        render_context
            .thin_lines
            .borrow_mut()
            .add_primitive(&vertices, &indices)
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

//
// INSTANCES
//

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

//
// Index Generating Functions
//

pub fn circle_indices_from_vertices(vertices: &[CircleVertex]) -> Result<Vec<u16>, Box<dyn Error>> {
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

/// Creates indices for a line single connected line from vertices
/// if loop_back is set to true, it will connect the first with the last vertex
pub fn line_indices_from_vertices(
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
    fn test_line_indices_from_vertices_loopback() {
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

        let indices = line_indices_from_vertices(&vertices, true).unwrap();

        assert_eq!(indices, vec![0, 1, 1, 2, 2, 3, 3, 0,])
    }

    #[test]
    fn test_line_indices_from_vertices_no_loopback() {
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

        let indices = line_indices_from_vertices(&vertices, false).unwrap();

        assert_eq!(indices, vec![0, 1, 1, 2, 2, 3])
    }
}
