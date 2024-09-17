use std::{f32::consts::TAU, thread::spawn};

use crate::render_context::QuadVertex;

use cgmath::{vec2, InnerSpace, MetricSpace, Vector2, Zero};
use itertools::Itertools;

const TIMESTEP: f32 = 1.0 / 60.0;
const MIN_DST: f32 = 0.1;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    position: Vector2<f32>,
    velocity: Vector2<f32>,
    acceleration: Vector2<f32>,
    mass: f32,
}

impl Particle {
    pub fn new(pos: Vector2<f32>, vel: Vector2<f32>, mass: f32) -> Particle {
        Particle {
            position: pos,
            velocity: vel,
            acceleration: Vector2::zero(),
            mass,
        }
    }

    pub fn update(&mut self, dt: f32) {
        const ONE_HALF: f32 = 1.0 / 2.0;
        self.position += self.velocity * dt + ONE_HALF * self.acceleration * dt * dt;
        self.velocity += self.acceleration * dt;
        self.acceleration = Vector2::zero();
    }
}

pub struct Simulation {
    pub bodies: Vec<Particle>,
}

impl Simulation {
    pub fn new(n: usize) -> Simulation {
        let mut bodies = Vec::with_capacity(n);
        let spawning_radius = 75.0;

        for _ in 0..n {
            let a = fastrand::f32() * TAU;
            let (sin, cos) = a.sin_cos();
            let r = (fastrand::f32() * 2.0 - 1.0) * spawning_radius;
            let pos = Vector2::new(cos, sin) * r;
            let vel = Vector2::new(sin, -cos) * r / spawning_radius;

            bodies.push(Particle::new(pos, vel, 1.0));
        }

        bodies.sort_by(|a, b| a.position.magnitude2().total_cmp(&b.position.magnitude2()));
        (0..n).for_each(|i| {
            let v = (i as f32 / bodies[i].position.magnitude()).sqrt();
            bodies[i].velocity *= v;
        });

        Simulation { bodies }
    }

    pub fn update(&mut self) {
        let n = self.bodies.len();
        (0..n)
            .cartesian_product(0..n)
            .filter(|(i, j)| i != j)
            .for_each(|(i, j)| {
                let r = self.bodies[j].position - self.bodies[i].position;
                let r_mag_squared = r.magnitude2();
                let r_unit = r.normalize();
                let mj = self.bodies[j].mass;
                self.bodies[i].acceleration += r_unit * mj / r_mag_squared.max(MIN_DST);
            });

        for i in 0..n {
            self.bodies[i].update(TIMESTEP);
        }
    }

    pub fn calculate_acc_at(&self, pos: Vector2<f32>) -> Vector2<f32> {
        let mut acc = vec2(0.0, 0.0);

        self.bodies.iter().for_each(|b| {
            let r = b.position - pos;
            let r_hat = r.normalize();
            let r_2 = r.magnitude2();
            let a = r_hat * b.mass / r_2;

            acc += a;
        });

        acc
    }
}

pub fn draw_body(body: Particle, radius: f32) -> [QuadVertex; 4] {
    let center: Vector2<f32> = body.position;
    let rad = Vector2::new(radius, -radius);

    let bl: Vector2<f32> = center - rad.xx();
    let br: Vector2<f32> = center - rad.yx();
    let tr: Vector2<f32> = center + rad.xx();
    let tl: Vector2<f32> = center + rad.yx();

    let color = [1.0, 1.0, 1.0, 1.0];

    [
        QuadVertex::new(bl, color, [-1.0, -1.0]),
        QuadVertex::new(br, color, [1.0, -1.0]),
        QuadVertex::new(tr, color, [1.0, 1.0]),
        QuadVertex::new(tl, color, [-1.0, 1.0]),
    ]
}
