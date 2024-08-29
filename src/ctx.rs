use cgmath::{Vector2, Vector4};

use crate::camera::Camera2D;
use crate::instances::circle::*;
use crate::instances::wireframe::*;
use crate::state::State;

pub struct Ctx {
    pub size: Vector2<u32>,
    pub circles: Vec<Circle>,
    pub circle_render: CircleRender,
    pub border: Wireframe,
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

        let circles = (0..5)
            .map(|_| Circle {
                position: Vector2::new(
                    (random() - 0.5) * size.x as f32,
                    (random() - 0.5) * size.y as f32,
                ),
                radius: random() * 100.0,
                color: Vector4::new(random(), random(), random(), 1.0),
            })
            .collect::<Vec<Circle>>();

        let border = Wireframe::new(
            [
                Vector2::new((size.x as f32), (size.y as f32)),
                Vector2::new(-(size.x as f32), (size.y as f32)),
                Vector2::new(-(size.x as f32), -(size.y as f32)),
                Vector2::new((size.x as f32), -(size.y as f32)),
            ],
            50.,
        );

        let circle_render = CircleRender::new(state, &camera);
        let mut wireframe_render = WireframeRender::new(state, &camera);

        wireframe_render.update_buffers(state, &[border.clone()]);

        Self {
            size,
            circles,
            circle_render,
            border,
            wireframe_render,
            camera,
        }
    }

    pub fn physics_process(&mut self, state: &State) {
        for c in &mut self.circles {
            c.position += Vector2::new(0.0, -2.0);
        }

        self.circle_render.update_instances(state, &self.circles);
    }
}
