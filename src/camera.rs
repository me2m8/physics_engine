use cgmath::{Vector2, Vector3};
use wgpu::util::DeviceExt;

use crate::state::State;

/// A 2d camera for use in a 2d scene
pub struct Camera2D {
    pub raw: RawCamera2D,
    pub uniform: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

/// A 3d camera for use in a 3d scene
pub struct Camera3D {
    /// The positoin of the camera
    eye: Vector3<f32>,
    /// The focal length.
    /// The distance between the eye and the screen projection
    fl: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct RawCamera2D {
    position: [f32; 2],
    width: f32,
    height: f32,
}

impl Camera2D {
    /// Creates a new [`Camera2D`] struct
    pub fn new(state: &State, viewport_size: Vector2<f32>) -> Self {
        let width = viewport_size.x;
        let height = viewport_size.y;

        let raw = RawCamera2D {
            position: [0., 0.],
            width,
            height,
        };

        let uniform = state
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera Uniform Buffer"),
                contents: bytemuck::cast_slice(&[raw]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            state
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Camera Bind Group Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let bind_group = state
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Camera Bind Group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 0,
                        size: None,
                    }),
                }],
            });

        Self {
            raw,
            uniform,
            bind_group_layout,
            bind_group,
        }
    }

    /// Changes the viewport size of the camera
    pub fn resize(&mut self, state: &State, width: f32, height: f32) {
        self.raw = RawCamera2D {
            position: self.raw.position,
            width,
            height,
        };
        self.update_bind_group(state);
    }

    /// Use this function to change the viewport size if you are scaling with the window
    pub fn scale_with_window(&mut self, state: &State, new_size: winit::dpi::PhysicalSize<u32>) {
        self.raw = RawCamera2D {
            position: self.raw.position,
            width: new_size.width as f32,
            height: new_size.height as f32,
        };

        self.update_bind_group(state);
    }

    /// Updates the position of the raw camera
    pub fn update_position(&mut self, state: &State, new_position: Vector2<f32>) {
        self.raw = RawCamera2D {
            position: new_position.into(),
            width: self.raw.width,
            height: self.raw.height,
        };

        self.update_bind_group(state);
    }

    /// Updates the bind_group and uniform buffer wih the current raw camera
    pub fn update_bind_group(&mut self, state: &State) {
        let uniform = state
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera Uniform Buffer"),
                contents: bytemuck::cast_slice(&[self.raw]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = state
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Camera Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 0,
                        size: None,
                    }),
                }],
            });

        self.uniform = uniform;
        self.bind_group = bind_group;
    }

    /// Returns the viewport width
    pub fn viewport_width(&self) -> f32 {
        self.raw.width
    }

    /// Returns the viewport height
    pub fn viewport_height(&self) -> f32 {
        self.raw.height
    }

    /// Returns the current in world position
    pub fn position(&self) -> Vector2<f32> {
        self.raw.position.into()
    }
}
