use cgmath::{vec2, Matrix4, Vector2, Vector3, VectorSpace};
use wgpu::{
    BindGroup, BindGroupLayout, BindingType, Buffer, BufferBindingType, BufferUsages, Device,
    ShaderStages,
};

pub struct CameraState<C>
where
    C: Camera + Sized,
{
    camera: C,
    uniform_buffer: Buffer,
    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,
}

impl<C> CameraState<C>
where
    C: Camera + Sized,
{
    pub fn new(device: &Device, viewport: Vector2<f32>) -> Self {
        let camera = C::new(viewport);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform"),
            size: dbg!(size_of::<C::Raw>() as u64).max(32),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        Self {
            camera,
            uniform_buffer,
            bind_group_layout,
            bind_group,
        }
    }

    pub fn viewport_size(&self) -> Vector2<f32> {
        self.camera.viewport_size()
    }

    pub fn to_raw(&self) -> C::Raw {
        self.camera.to_raw()
    }
    pub fn uniform_buffer(&self) -> &Buffer {
        &self.uniform_buffer
    }
    pub fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }
    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}

pub trait Camera {
    type Raw: bytemuck::Pod + bytemuck::Zeroable;
    type NDVector: VectorSpace;

    fn new(viewport: Vector2<f32>) -> Self;
    fn to_raw(&self) -> Self::Raw;
    fn position(&self) -> Self::NDVector;
    fn viewport_size(&self) -> Vector2<f32>;
    fn translate(&mut self, position: Self::NDVector);
    fn change_viewport(&mut self, viewport: Vector2<f32>);
}

//
// Camera2D
//

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RawCamera2D {
    matrix: [[f32; 4]; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct Camera2D {
    position: Vector2<f32>,
    viewport: Vector2<f32>,
}
impl Camera for Camera2D {
    type Raw = RawCamera2D;
    type NDVector = Vector2<f32>;

    /// Creates a new 2d camera
    fn new(viewport: Vector2<f32>) -> Self {
        Self {
            position: vec2(0.0, 0.0),
            viewport,
        }
    }

    /// returns the camera data for buffer binding
    fn to_raw(&self) -> RawCamera2D {
        let width = self.viewport.x;
        let height = self.viewport.y;

        let far = 1.0;
        let near = 0.0;

        RawCamera2D {
            #[rustfmt::skip]
            matrix: Matrix4::new(
                2.0 / width, 0.0,           0.0,                  0.0,
                0.0,         2.0 / height,  0.0,                  0.0,
                0.0,         0.0,          -2.0 / (far - near),   0.0,
                0.0,         0.0,           0.0,                  1.0,
            ).into(),
        }
    }

    fn position(&self) -> Self::NDVector {
        self.position
    }

    fn viewport_size(&self) -> Vector2<f32> {
        self.viewport
    }

    fn translate(&mut self, position: Self::NDVector) {
        self.position = position;
    }

    fn change_viewport(&mut self, viewport: Vector2<f32>) {
        self.viewport = viewport;
    }
}

fn orthographic_projection(
    width: f32,
    height: f32,
    top_left_x: f32,
    top_left_y: f32,
) -> Matrix4<f32> {
    let near = 0.0;
    let far = 1.0;
    let right = top_left_x + width;
    let left = top_left_y;
    let bottom = top_left_y - height;
    let top = top_left_y;

    Matrix4::new(
        2.0 / (right - left),
        0.0,
        0.0,
        -(right + left) / (right - left),
        0.0,
        2.0 / (top - bottom),
        0.0,
        -(top + bottom) / (top - bottom),
        0.0,
        0.0,
        -2.0 / (far - near),
        -(far + near) / (far - near),
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

//
// Camera3D
//

#[allow(unused)]
#[derive(Clone, Copy, Debug)]
pub struct Camera3D {
    eye: Vector3<f32>,
    rotations: Vector3<f32>,
    viewport: Vector2<f32>,
    fov: f32,
}

#[allow(unused)]
#[derive(Clone, Copy, Debug)]
pub struct RawCamera3D {
    projection_matrix: Matrix4<f32>,
}

//
// a = h/w
//
// f = 1 / tan(fov / 2)
//
// zfar is the farthest we can render
// znear is the nearest we can render
//
// lambda = zfar(1 - znear) / (zfar - znear)
// lambda is the z scaling factor

//
// [afx, fy, lambda*z - lambda * znear]
