#![allow(unused)]
use core::num;
use std::{
    borrow::{BorrowMut, Cow},
    cell::RefCell,
    collections::{HashMap, VecDeque},
    error::Error,
    fmt::Debug,
    marker::PhantomData,
    num::NonZero,
};

use bytemuck::bytes_of;
use cgmath::{vec2, vec3, vec4, Vector2, Vector3, Vector4};
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
    pad_bytes, MAX_CIRCLES, SAMPLE_COUNT,
};

use crate::shaders::{make_pipelines, PipelineType};

pub struct RenderContext<C>
where
    C: Camera + Sized,
{
    pipelines: HashMap<PipelineType, wgpu::RenderPipeline>,
    config: wgpu::SurfaceConfiguration,
    camera: CameraState<C>,

    circles: DrawPrimitive<CircleVertex, CirclePrimitive>,
}

impl<C> RenderContext<C>
where
    C: Camera + Sized,
{
    pub fn new(device: &Device, config: &SurfaceConfiguration) -> Self {
        let float_width = config.width as f32;
        let float_height = config.height as f32;

        dbg!(float_width, float_height);

        // This scales the viewport such that the width becomes this amount in pixels
        let viewport_scale = crate::VIEWPORT_SCALE;

        let viewport = vec2(float_width, float_height) / float_width * viewport_scale;
        dbg!(viewport);
        let camera = CameraState::new(device, viewport);

        let pipelines = make_pipelines(device, config, &camera);

        let circles = DrawPrimitive::new(device, crate::MAX_CIRCLES);

        Self {
            camera,
            pipelines,
            config: config.clone(),

            circles,
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

        // Use .chain() to chain the shapes iterators
        for data in self.circles.shapes.borrow_mut() {
            self.circles.populate_buffers(queue, data);

            {}
        }

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
// SHAPES QUEUE
//

/// Iterator that returns all data up to the set limit.
pub struct ShapeQueue<V, S>
where
    V: Vertex + Sized + bytemuck::Pod,
    S: PrimitiveShape<Vert = V> + Sized,
{
    shapes: VecDeque<S>,
    limit: usize,
}

impl<V, S> ShapeQueue<V, S>
where
    V: Vertex + Sized + bytemuck::Pod,
    S: PrimitiveShape<Vert = V> + Sized,
{
    const V_SIZE: usize = size_of::<V>();

    fn new(limit: usize) -> Self {
        Self {
            limit,
            ..Default::default()
        }
    }

    /// Pushes a shape to the queue.
    fn add_shape(&mut self, shape: S) {
        self.shapes.push_back(shape);
    }

    /// Returns the data of the shapes inside the queue. The data will at most encapsule the number
    /// of shapes given in [limit].
    fn get_to_limit(&mut self) -> Option<(Vec<V>, Vec<u16>)> {
        if self.shapes.is_empty() {
            return None;
        }

        let n_shapes = self.shapes.len().min(self.limit);

        let mut vertices: Vec<V> = Vec::with_capacity(n_shapes * S::N_VERTICES);
        let mut indices: Vec<u16> = Vec::with_capacity(n_shapes * S::N_INDICES);

        let data = (0..n_shapes).for_each(|_| {
            // Pop the shape from the queue
            let s = self.shapes.pop_front().unwrap();

            // Get the vertex and index data of the shape
            let (mut v, mut i) = s.to_data();

            // Offset the index by the number of vertices already in the buffer
            let n_v = vertices.len() as u16;
            i.iter_mut().for_each(|i| *i += n_v);

            // Append the vertices and indices to the buffer
            vertices.append(&mut v);
            indices.append(&mut i);
        });

        Some((vertices, indices))
    }
}

impl<V, S> Default for ShapeQueue<V, S>
where
    V: Vertex + Sized + bytemuck::Pod,
    S: PrimitiveShape<Vert = V> + Sized,
{
    fn default() -> Self {
        Self {
            shapes: Default::default(),
            limit: 0,
        }
    }
}

impl<V, S> Iterator for ShapeQueue<V, S>
where
    V: Vertex + Sized + bytemuck::Pod,
    S: PrimitiveShape<Vert = V> + Sized,
{
    type Item = (Vec<V>, Vec<u16>);

    fn next(&mut self) -> Option<Self::Item> {
        self.get_to_limit()
    }
}

//
// PRIMITIVE SHAPES
//

pub trait PrimitiveShape: Sized {
    type Vert: Vertex + Sized + bytemuck::Pod;

    const PIPELINE: PipelineType;
    const N_VERTICES: usize;
    const N_INDICES: usize;

    fn to_data(&self) -> (Vec<Self::Vert>, Vec<u16>);
}

pub struct CirclePrimitive {
    radius: f32,
    position: Vector2<f32>,
    color: Vector4<f32>,
}

impl PrimitiveShape for CirclePrimitive {
    type Vert = CircleVertex;

    const PIPELINE: PipelineType = PipelineType::CircleFill;
    const N_VERTICES: usize = 4;
    const N_INDICES: usize = 6;

    fn to_data(&self) -> (Vec<Self::Vert>, Vec<u16>) {
        let p = self.position;
        let r = self.radius;
        let color = self.color;

        let center = vec4(p.x, p.y, 0.0, 1.0);
        let rad = vec3(r, -r, 0.0);

        let bl = center - rad.xxzz();
        let br = center - rad.yxzz();
        let tr = center + rad.xxzz();
        let tl = center + rad.yxzz();

        #[rustfmt::skip]
        let vertices = vec![
            CircleVertex { p: bl.into(), c: color.into(), fc: [-1., -1.]},
            CircleVertex { p: br.into(), c: color.into(), fc: [-1.,  1.]},
            CircleVertex { p: tr.into(), c: color.into(), fc: [ 1.,  1.]},
            CircleVertex { p: tl.into(), c: color.into(), fc: [ 1., -1.]},
        ];
        let indices = vec![0, 1, 2, 2, 3, 0];

        (vertices, indices)
    }
}

//
// PRIMITIVES
//

pub struct DrawPrimitive<V, S>
where
    V: Vertex + Sized + bytemuck::Pod,
    S: PrimitiveShape<Vert = V> + Sized,
{
    shapes: RefCell<ShapeQueue<V, S>>,

    ib: RefCell<Buffer>,
    vb: RefCell<Buffer>,
}

impl<V, S> DrawPrimitive<V, S>
where
    V: Vertex + Sized + bytemuck::Pod + Debug,
    S: PrimitiveShape<Vert = V> + Sized,
{
    fn new(device: &Device, max_prim: usize) -> Self {
        let vb = RefCell::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (size_of::<V>() * max_prim * S::N_VERTICES) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let ib = RefCell::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (size_of::<u16>() * max_prim * S::N_INDICES) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        Self {
            shapes: RefCell::new(ShapeQueue::new(max_prim)),

            vb,
            ib,
        }
    }

    #[inline]
    pub fn add_shape(&mut self, shape: S) {
        self.shapes.borrow_mut().add_shape(shape);
    }

    /// Populates the buffers with data. If there is no data, this function just returns
    pub fn populate_buffers(self, queue: &Queue, data: (Vec<V>, Vec<u16>)) {
        let (v, i) = data;

        let mut vb_bytes = bytemuck::cast_slice::<V, u8>(&v).to_vec();
        let mut ib_bytes = bytemuck::cast_slice::<u16, u8>(&i).to_vec();

        let new_vb_size = pad_bytes(&mut vb_bytes, COPY_BUFFER_ALIGNMENT as usize) as u64;
        let new_ib_size = pad_bytes(&mut ib_bytes, COPY_BUFFER_ALIGNMENT as usize) as u64;

        let vb = self.vb.borrow_mut();
        let ib = self.ib.borrow_mut();

        let mut vb_view =
            queue.write_buffer_with(&vb, 0, unsafe { BufferSize::new_unchecked(new_vb_size) });
        let mut ib_view =
            queue.write_buffer_with(&ib, 0, unsafe { BufferSize::new_unchecked(new_ib_size) });

        vb_view.unwrap().as_mut().copy_from_slice(&vb_bytes);
        ib_view.unwrap().as_mut().copy_from_slice(&ib_bytes);
    }

    /// Renders the primitive on the given render pass
    //pub fn render_primitive(&mut self, render_pass: &mut RenderPass, pipeline: &RenderPipeline) {
    //    if !self.in_use {
    //        return;
    //    }
    //
    //    info!("Drawing primitive");
    //    render_pass.set_pipeline(pipeline);
    //
    //    render_pass.set_vertex_buffer(0, self.vb().slice(..));
    //    render_pass.set_index_buffer(self.ib().slice(..), wgpu::IndexFormat::Uint16);
    //
    //    render_pass.draw_indexed(0..self.num_indices(), 0, 0..1);
    //
    //    self.in_use = false;
    //}

    pub fn ib(&self) -> &Buffer {
        &self.ib
    }
    pub fn vb(&self) -> &Buffer {
        &self.vb
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
