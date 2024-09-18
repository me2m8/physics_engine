#![allow(unused)]
use core::num;
use std::{borrow::Cow, cell::RefCell, collections::HashMap, error::Error, num::NonZero};

use cgmath::{vec2, Vector2, Vector3, Vector4};
use itertools::Itertools;
use wgpu::{
    include_wgsl, util::{DeviceExt, RenderEncoder}, vertex_attr_array, BlendComponent, BlendState, Buffer, BufferAddress, BufferUsages, ColorWrites, Device, FragmentState, FrontFace, PipelineCompilationOptions, PrimitiveState, PrimitiveTopology, Queue, QueueWriteBufferView, RenderPipeline, ShaderModule, Surface, SurfaceConfiguration, VertexAttribute, VertexBufferLayout
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

    circles: RefCell<Primitive<CircleVertex>>,
    thin_line: RefCell<Primitive<LineVertex>>,
    arrow: RefCell<Primitive<ArrowVertex>>,
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
            size: (size_of::<[CircleVertex; 4]>() * crate::MAX_QUADS) as u64,
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
            size: (size_of::<[ArrowVertex; 4]>() * crate::MAX_ARROWS) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let arrow_ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (size_of::<u16>() * crate::MAX_ARROWS * 9) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            camera,
            pipelines,
        }
    }

    /// This function currently does nothing and just panics
    pub fn begin_scene(&mut self, queue: &Queue) {
        panic!("Currently does nothing")
    }

    pub fn present_scene(&mut self, queue: &Queue, device: &Device, surface: &Surface) {
        // writing the buffers with stored vertices

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
            // render_pass.set_pipeline(self.pipeline(PipelineType::ZeroWidthLines));

            // render_pass.set_vertex_buffer(0, self.line_vb.slice(..));
            // render_pass.set_index_buffer(self.line_ib.slice(..), wgpu::IndexFormat::Uint16);

            // render_pass.draw_indexed(0..line_indicies, 0, 0..1);

            // // Draw the Circles
            // render_pass.set_pipeline(self.pipeline(PipelineType::CircleFill));

            // render_pass.set_vertex_buffer(0, self.circle_vb.slice(..));
            // render_pass.set_index_buffer(self.circle_ib.slice(..), wgpu::IndexFormat::Uint16);

            // render_pass.draw_indexed(0..circle_indicies, 0, 0..1);
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

pub struct Primitive<V: Vertex + Sized> {
    vertices: Vec<V>,
    indicies: Vec<u16>,
    ib: Buffer,
    vb: Buffer,
}

impl<T: Vertex + Sized> Primitive<T> {
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
            indicies: Default::default(),

            ib,
            vb,
        }
    }
    pub fn add_vertices(&mut self, vertices: &[T]) {
        self.vertices.extend(vertices.iter());
    }
    pub fn add_indicies(&mut self, indicies: &[u16]) {
        let num_vertices = self.vertices.len() as u16;
        self.indicies
            .extend(indicies.iter().map(|i| i + num_vertices));
    }

    pub fn add_primitive(&mut self, vertices: &[T], indicies: &[u16]) {
        self.add_indicies(indicies);
        self.add_vertices(vertices);
    }

    pub fn ib(&self) -> &Buffer {
        &self.ib
    }
    pub fn vb(&self) -> &Buffer {
        &self.vb
    }
}

mod shapes {
    use std::borrow::BorrowMut;

    use cgmath::{vec3, vec4, Angle, Rad};

    use crate::camera::Camera2D;

    use super::*;

    /// Draws a circle at a given position with a given radius and color
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

        render_context
            .circles
            .borrow_mut()
            .add_primitive(&vertices, &indicies);
    }

    pub fn construct_arrow_2d(
        render_context: &RenderContext<Camera2D>,
        length: f32,
        line_thickness: f32,
        point_length: f32,
        point_width: f32,
    ) -> impl Fn(Vector2<f32>, Rad<f32>) + '_ {
        |p: Vector2<f32>, dir: Rad<f32>| move {
            draw_arrow_2d(
                render_context,
                p,
                dir,
                length,
                line_thickness,
                point_length,
                point_width,
            );
        }
    }

    ///
    pub fn draw_arrow_2d(
        render_context: &RenderContext<Camera2D>,
        p: Vector2<f32>,
        dir: Rad<f32>,
        length: f32,
        line_thickness: f32,
        point_length: f32,
        point_width: f32,
    ) {
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
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub frag_coord: [f32; 2],
}
impl Vertex for ArrowVertex {
    const ATTRIBS: &'static [VertexAttribute] =
        &vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x2];
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

pub fn circle_indicies_from_verticies(
    vertices: &[CircleVertex],
) -> Result<Vec<u16>, Box<dyn Error>> {
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
