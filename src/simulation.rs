use cgmath::{vec2, vec4, Vector2};
use itertools::Itertools;

use crate::render_context::{LineVertex, QuadVertex};

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vector2<f32>,
    pub radius: f32,
}

impl Particle {
    pub fn to_vertices(&self) -> [QuadVertex; 4] {
        use crate::PARTICLE_COLOR;

        let rad = self.radius;

        let center = vec4(self.position.x, self.position.y, 0.0, 1.0);
        let bl = vec4(-rad, -rad, 0.0, 0.0);
        let br = vec4( rad, -rad, 0.0, 0.0);
        let tr = vec4( rad,  rad, 0.0, 0.0);
        let tl = vec4(-rad,  rad, 0.0, 0.0);

        [
            QuadVertex {
                position: (center + bl).into(),
                color: PARTICLE_COLOR,
                frag_coord: [-1.0, -1.0],
            },
            QuadVertex {
                position: (center + br).into(),
                color: PARTICLE_COLOR,
                frag_coord: [1.0, -1.0],
            },
            QuadVertex {
                position: (center + tr).into(),
                color: PARTICLE_COLOR,
                frag_coord: [1.0, 1.0],
            },
            QuadVertex {
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
        let mut particles = (0..100)
            .map(|_| Particle {
                position: vec2(
                    dbg!(rand::random::<f32>() - 0.5) * BOUNDARY_WIDTH,
                    dbg!(rand::random::<f32>() - 0.5) * BOUNDARY_HEIGHT,
                ),
                radius: 10.0,
            })
            .collect_vec();

        particles.push(Particle { position: vec2(0.0, 0.0), radius: 10.0 });

        Self { particles }
    }

    pub fn particles_to_circle_vertices(&self) -> Vec<QuadVertex> {
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

pub fn simulation_border() -> [LineVertex; 4] {
    let hw = BOUNDARY_WIDTH / 2.0;
    let hh = BOUNDARY_HEIGHT / 2.0;

    let bl = vec4(-hw, -hh, 0.0, 1.0);
    let br = vec4( hw, -hh, 0.0, 1.0);
    let tr = vec4( hw,  hh, 0.0, 1.0);
    let tl = vec4(-hw,  hh, 0.0, 1.0);

    [
        LineVertex { position: bl.into(), color: [1.0, 1.0, 1.0, 1.0] },
        LineVertex { position: br.into(), color: [1.0, 1.0, 1.0, 1.0] },
        LineVertex { position: tr.into(), color: [1.0, 1.0, 1.0, 1.0] },
        LineVertex { position: tl.into(), color: [1.0, 1.0, 1.0, 1.0] },
    ]
}
