use wgpu::util::DeviceExt;

use crate::state::State;

pub struct Camera2D {
    pub raw: RawCamera2D,
    pub uniform: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct RawCamera2D {
    width: f32,
    height: f32,
}

impl Camera2D {
    pub fn new(state: &State) -> Self {
        let width = state.config().width as f32;
        let height = state.config().height as f32;

        let raw = RawCamera2D {
            width,
            height,
        };

        let uniform = state.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Uniform Buffer"),
                contents: bytemuck::cast_slice(&[raw]),
                usage: wgpu::BufferUsages::UNIFORM,
            }
        );

        let bind_group_layout = state.device().create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
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
            }
        );

        let bind_group = state.device().create_bind_group(
            &wgpu::BindGroupDescriptor {
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
            }
        );

        Self {
            raw,
            uniform,
            bind_group_layout,
            bind_group,
        }
    }

    pub fn resize(&mut self,state: &State, width: f32, height: f32) {
        let raw = RawCamera2D {
            width,
            height,
        };

        let uniform = state.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Uniform Buffer"),
                contents: bytemuck::cast_slice(&[raw]),
                usage: wgpu::BufferUsages::UNIFORM,
            }
        );

        let bind_group = state.device().create_bind_group(
            &wgpu::BindGroupDescriptor {
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
            }
        );

        self.raw = raw;
        self.uniform = uniform;
        self.bind_group = bind_group;

    }

    pub fn scale_with_view(&mut self, state: &State, new_size: winit::dpi::PhysicalSize<u32>) {
        let raw = RawCamera2D {
            width: new_size.width as f32,
            height: new_size.height as f32,
        };

        let uniform = state.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Uniform Buffer"),
                contents: bytemuck::cast_slice(&[raw]),
                usage: wgpu::BufferUsages::UNIFORM,
            }
        );

        let bind_group = state.device().create_bind_group(
            &wgpu::BindGroupDescriptor {
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
            }
        );

        self.raw = raw;
        self.uniform = uniform;
        self.bind_group = bind_group;

    }

    pub fn width(&self) -> f32 {
        self.raw.width
    }

    pub fn height(&self) -> f32 {
        self.raw.height
    }
}
