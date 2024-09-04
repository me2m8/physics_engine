use cgmath::{vec4, Vector3, Vector4};

use crate::render_context::Vertex;
use std::f32::consts::FRAC_1_SQRT_2;

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vector3<f32>,
    pub radius: f32,
}

impl Particle {
    pub fn to_vertices(&self) -> [Vertex; 4] {
        let rad_root_2 = FRAC_1_SQRT_2 * self.radius;

        let center = vec4(self.position.x, self.position.y, self.position.z, 1.0);
        let bl = vec4(-rad_root_2, -rad_root_2, 0.0, 1.0);
        let br = vec4( rad_root_2, -rad_root_2, 0.0, 1.0);
        let tr = vec4( rad_root_2,  rad_root_2, 0.0, 1.0);
        let tl = vec4(-rad_root_2,  rad_root_2, 0.0, 1.0);

        [
            Vertex { position: (center + bl).into(), color: [0.0, 0.5, 1.0, 1.0], frag_coord: [-1.0, -1.0] },
            Vertex { position: (center + br).into(), color: [0.5, 0.5, 1.0, 1.0], frag_coord: [ 1.0, -1.0] },
            Vertex { position: (center + tr).into(), color: [0.0, 0.5, 1.0, 1.0], frag_coord: [ 1.0,  1.0] },
            Vertex { position: (center + tl).into(), color: [0.5, 0.5, 1.0, 1.0], frag_coord: [-1.0,  1.0] }
        ]
    }
}
