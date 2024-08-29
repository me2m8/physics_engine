use cgmath::{Vector2, Vector4};

use crate::camera::Camera2D;
use crate::instances::wireframe::*;
use crate::state::State;
use crate::instances::circle::*;

pub struct Ctx {
    pub size: Vector2<u32>,
    pub circles: Vec<Circle>,
    pub circle_render: CircleRender,
    pub wireframes: Vec<Wireframe>,
    pub wireframe_render: WireframeRender,
    pub camera: Camera2D,
}

pub struct LineRender {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub num_indicies: u32,
    pub num_instances: u32,
}

fn random() -> f32 {
    rand::random::<f32>()
}

impl Ctx {
    pub fn new(state: &State, size: Vector2<u32>) -> Self {

        let camera = Camera2D::new(state);

        let circles = (0..5).map(|_| Circle {
            position: Vector2::new((random() - 0.5) * size.x as f32, (random() - 0.5) * size.y as f32),
            radius: random() * 100.0,
            color: Vector4::new(random(), random(), random(), 1.0),
        }).collect::<Vec<Circle>>();

        let wireframes = vec![
            Wireframe::new([
                Vector2::new( (size.x as f32) / 2.,  (size.y as f32) / 2.),
                Vector2::new(-(size.x as f32) / 2.,  (size.y as f32) / 2.),
                Vector2::new(-(size.x as f32) / 2., -(size.y as f32) / 2.),
                Vector2::new( (size.x as f32) / 2., -(size.y as f32) / 2.),
            ], 50.)
        ];

        let circle_render = CircleRender::new(state, &camera);
        let mut wireframe_render = WireframeRender::new(state, &camera);

        wireframe_render.update_buffers(state, &wireframes);

        Self {
            size,
            circles,
            circle_render,
            wireframes,
            wireframe_render,
            camera,
        }
    }

    pub fn physics_process(&mut self, state: &State) {
        self.circles.push(Circle {
            position: Vector2::new((random() * 2.0 - 1.0) * state.config().width as f32, (random() * 2. - 1.) * state.config().height as f32),
            radius: random() * 100.0,
            color: Vector4::new(random(), random(), random(), 1.0),
        });

        self.circle_render.update_instances(state, &self.circles);
    }
}
