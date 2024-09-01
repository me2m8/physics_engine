use cgmath::{vec2, vec4};
use itertools::Itertools;

use crate::{instances::circle::CircleRender, particle::Particle, state::State};

pub struct SimulationCtx {
    particles: Vec<Particle>,
}

fn random(lower_bound: f32, upper_bound: f32) -> f32 {
    lower_bound + rand::random::<f32>() * (upper_bound - lower_bound)
}

impl SimulationCtx {
    pub fn new() -> Self {
        let grid_size = 27;

        let particles = (0..grid_size)
            .cartesian_product(0..grid_size)
            .map(|(i, j)| {
                let (i, j) = (i as f32, j as f32);
                let grid_size = grid_size as f32;

                let particle_radius = random(5., 10.);
                let particle_mass = particle_radius;

                let position = vec2(
                    (i - grid_size / 2.) * 50. * 2.5,
                    (j - grid_size / 2.) * 50. * 2.5,
                );
                let velocity = vec2(random(-400., 400.), random(400., 800.));
                let color = vec4(
                    1. - i / grid_size,
                    1. - j / grid_size,
                    (i + j) / (grid_size * 2.),
                    1.0,
                );

                Particle::new(particle_radius, particle_mass, position, velocity, color)
            })
            .collect_vec();

        Self { particles }
    }

    /// Send the particles to the circle renderer
    pub fn draw_particles(&self, state: &State, circle_renderer: &mut CircleRender) {
        let circles = self.particles.iter().map(|p| p.circle()).collect_vec();

        circle_renderer.update_instances(state, &circles);
    }

    pub fn update(&mut self, dt: f32) {
        self.particles.iter_mut().for_each(|p| p.update(dt));

        (0..self.particles.len()).for_each(|i| {
            (0..self.particles.len() - i - 1).for_each(|j| {
                let (s1, s2) = self.particles.split_at_mut(i + 1);
                s1[i].handle_particle_collision(&mut s2[j]);
            })
        })
    }
}

impl Default for SimulationCtx {
    fn default() -> Self {
        Self::new()
    }
}
