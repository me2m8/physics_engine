use cgmath::{InnerSpace, Vector2, Zero};
use wgpu::util::DeviceExt;

use crate::state::State;

pub struct Camera2D {
    pub raw: RawCamera2D,
    pub uniform: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
        controller: CameraController,
}

#[derive(Default)]
struct CameraController {
    up_arrow_pressed: bool,
    down_arrow_pressed: bool,
    left_arrow_pressed: bool,
    right_arrow_pressed: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct RawCamera2D {
    position: [f32; 2],
    width: f32,
    height: f32,
}

impl Camera2D {
    pub fn new(state: &State) -> Self {
        let surface_width = state.config().width;
        let surface_height = state.config().height;

        let width = state.config().width as f32;
        let height = state.config().height as f32;

        println!("surface width and height: {surface_width:?}, {surface_height:?}");
        println!("camera width and height: {width:?}, {height:?}");

        let raw = RawCamera2D {
            position: [0., 0.],
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

        let controller = CameraController::default(); 

        Self {
            raw,
            uniform,
            bind_group_layout,
            bind_group,
            controller,
        }
    }

    pub fn resize(&mut self,state: &State, width: f32, height: f32) {
        self.raw = RawCamera2D {
            position: self.raw.position,
            width,
            height,
        };

        self.update_bind_group(state);
    }

    pub fn scale_with_view(&mut self, state: &State, new_size: winit::dpi::PhysicalSize<u32>) {
        self.raw = RawCamera2D {
            position: self.raw.position,
            width: new_size.width as f32,
            height: new_size.height as f32,
        };

        self.update_bind_group(state);
    }

    pub fn update_position(&mut self, state: &State, new_position: Vector2<f32>) {
        self.raw = RawCamera2D {
            position: new_position.into(),
            width: self.raw.width,
            height: self.raw.height
        };

        self.update_bind_group(state);
    }

    pub fn update_bind_group(&mut self, state: &State) {

        let uniform = state.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Uniform Buffer"),
                contents: bytemuck::cast_slice(&[self.raw]),
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

        self.uniform = uniform;
        self.bind_group = bind_group;
    }

    pub fn process_input(&mut self, event: winit::event::KeyEvent) {
        match event {
            winit::event::KeyEvent {
                physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowUp),
                state,
                ..
            } => self.controller.up_arrow_pressed = state == winit::event::ElementState::Pressed,
            winit::event::KeyEvent {
                physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowDown),
                state,
                ..
            } => self.controller.down_arrow_pressed = state == winit::event::ElementState::Pressed,
            winit::event::KeyEvent {
                physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowLeft),
                state,
                ..
            } => self.controller.left_arrow_pressed = state == winit::event::ElementState::Pressed,
            winit::event::KeyEvent {
                physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowRight),
                state,
                ..
            } => self.controller.right_arrow_pressed = state == winit::event::ElementState::Pressed,
            _ => {}
        }
    }

    pub fn update(&mut self, state: &State) {
        let mut vel = Vector2::<f32>::zero();

        const SPEED: f32 = 20.0;

        vel.x = self.controller.right_arrow_pressed as u32 as f32 - self.controller.left_arrow_pressed as u32 as f32;
        vel.y = self.controller.up_arrow_pressed as u32 as f32 - self.controller.down_arrow_pressed as u32 as f32;

        if vel != Vector2::<f32>::zero() {
            let new_pos = self.position() + vel.normalize() * SPEED;

            self.update_position(state, new_pos);
        }
    }

    pub fn width(&self) -> f32 {
        self.raw.width
    }

    pub fn height(&self) -> f32 {
        self.raw.height
    }

    pub fn position(&self) -> Vector2<f32> {
        self.raw.position.into()
    }
}
