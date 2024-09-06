use cgmath::{vec2, vec4, Vector2};
use itertools::Itertools;

use crate::render_context::Vertex;
use std::f32::consts::SQRT_2;

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vector2<f32>,
    pub radius: f32,
}

impl Particle {
    pub fn to_vertices(&self) -> [Vertex; 4] {
        use crate::PARTICLE_COLOR;

        let rad_root_2 = SQRT_2 * self.radius;

        let center = vec4(self.position.x, self.position.y, 0.0, 1.0);
        let bl = vec4(-rad_root_2, -rad_root_2, 0.0, 1.0);
        let br = vec4(rad_root_2, -rad_root_2, 0.0, 1.0);
        let tr = vec4(rad_root_2, rad_root_2, 0.0, 1.0);
        let tl = vec4(-rad_root_2, rad_root_2, 0.0, 1.0);

        [
            Vertex {
                position: (center + bl).into(),
                color: PARTICLE_COLOR,
                frag_coord: [-1.0, -1.0],
            },
            Vertex {
                position: (center + br).into(),
                color: PARTICLE_COLOR,
                frag_coord: [1.0, -1.0],
            },
            Vertex {
                position: (center + tr).into(),
                color: PARTICLE_COLOR,
                frag_coord: [1.0, 1.0],
            },
            Vertex {
                position: (center + tl).into(),
                color: PARTICLE_COLOR,
                frag_coord: [-1.0, 1.0],
            },
        ]
    }
}

pub struct SimulationContext {
    particles: Vec<Particle>,
}

const BOUNDARY_WIDTH: f32 = 1900.0;
const BOUNDARY_HEIGHT: f32 = 1060.0;

impl SimulationContext {
    pub fn new() -> Self {
        let particles = (0..100)
            .map(|_| Particle {
                position: vec2(
                    (rand::random::<f32>() * 2.0 - 1.0) * BOUNDARY_WIDTH / 2.0,
                    (rand::random::<f32>() * 2.0 - 1.0) * BOUNDARY_HEIGHT / 2.0,
                ),
                radius: 10.0,
            })
            .collect_vec();

        Self { particles }
    }

    pub fn particles_to_circle_vertices(&self) -> Vec<Vertex> {
        self.particles
            .iter()
            .map(|p| p.to_vertices())
            .collect_vec()
            .concat()
    }
}

impl Default for SimulationContext {
    fn default() -> Self {
        Self::new()
    }
}

pub fn simulation_border() -> [Vertex; 4] {
    use std::f32::consts::FRAC_1_SQRT_2;

    let bl = vec4(-BOUNDARY_WIDTH * FRAC_1_SQRT_2, -BOUNDARY_HEIGHT * FRAC_1_SQRT_2, 0.0, 1.0);
    let br = vec4( BOUNDARY_WIDTH * FRAC_1_SQRT_2, -BOUNDARY_HEIGHT * FRAC_1_SQRT_2, 0.0, 1.0);
    let tr = vec4( BOUNDARY_WIDTH * FRAC_1_SQRT_2,  BOUNDARY_HEIGHT * FRAC_1_SQRT_2, 0.0, 1.0);
    let tl = vec4(-BOUNDARY_WIDTH * FRAC_1_SQRT_2,  BOUNDARY_HEIGHT * FRAC_1_SQRT_2, 0.0, 1.0);

    [
        Vertex { position: bl.into(), color: [1.0, 1.0, 1.0, 1.0], frag_coord: [0.0, 0.0] },
        Vertex { position: br.into(), color: [1.0, 1.0, 1.0, 1.0], frag_coord: [0.0, 0.0] },
        Vertex { position: tr.into(), color: [1.0, 1.0, 1.0, 1.0], frag_coord: [0.0, 0.0] },
        Vertex { position: tl.into(), color: [1.0, 1.0, 1.0, 1.0], frag_coord: [0.0, 0.0] },
    ]
}
