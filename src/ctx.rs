use cgmath::Vector2;
use itertools::Itertools;

use crate::camera::Camera2D;
use crate::instances::circle::*;
use crate::instances::polygon_frame::*;
use crate::particle::Particle;
use crate::state::State;

pub struct Ctx {
    pub size: Vector2<f32>,
    pub particles: Vec<Particle>,
    pub circle_render: CircleRender,
    pub border: PolygonFrame,
    pub polygon_frame_render: PolygonFrameRender,
    pub camera: Camera2D,

    last_frame: std::time::Instant,
    dt: std::time::Duration,
}

fn random() -> f32 {
    rand::random::<f32>()
}

impl Ctx {

    pub fn new(state: &State, size: Vector2<f32>) -> Self {
        let camera = Camera2D::new(state, size);

        let grid_size = 10;

        let particles =
            (0..grid_size)
                .cartesian_product(0..grid_size)
                .fold(vec![], |mut acc, (i, j)| {
                    acc.push(Particle::new(
                        20.,
                        10.,
                        Vector2::new(
                            50. * (i - grid_size / 2) as f32,
                            50. * (j - grid_size / 2) as f32,
                        ),
                        None,
                    ));
                    acc
                });

        let half_bw = crate::BORDER_WIDTH / 2.;
        let half_width = size.x / 2.;
        let half_height = size.y / 2.;

        let border = PolygonFrame::new(
            [
                Vector2::new(half_width - half_bw, half_height - half_bw),
                Vector2::new(-(half_width - half_bw), half_height - half_bw),
                Vector2::new(-(half_width - half_bw), -(half_height - half_bw)),
                Vector2::new(half_width - half_bw, -(half_height - half_bw)),
            ],
            crate::BORDER_WIDTH,
        );

        let circle_render = CircleRender::new(state, &camera);
        let mut polygon_frame_render = PolygonFrameRender::new(state, &camera);

        polygon_frame_render.update_buffers(state, &[border.clone()]);

        let dt = std::time::Duration::ZERO;
        let last_frame = std::time::Instant::now();

        Self {
            size,
            particles,
            circle_render,
            border,
            polygon_frame_render,
            camera,
            last_frame,
            dt,
        }
    }

    pub fn update_dt(&mut self) {
        let now = std::time::Instant::now();

        self.dt = now - self.last_frame;
        self.last_frame = now;
    }

    pub const fn dt(&self) -> std::time::Duration {
        self.dt
    }

    pub fn physics_process(&mut self, state: &State) {
        let dt = self.dt;
        self.particles
            .iter_mut()
            .for_each(|p| p.physics_update(state, dt));

        self.circle_render.update_instances(
            state,
            &self
                .particles
                .iter()
                .map(|p| p.circle())
                .collect::<Vec<_>>(),
        );
    }
}
