use std::{f32::consts::PI, sync::Mutex, time::Instant};

use cgmath::{vec2, InnerSpace, Vector2, Zero};
use itertools::Itertools;
use lazy_static::lazy_static;
use rand::{random, seq::IteratorRandom};

use rayon::prelude::*;
use tracing_subscriber::fmt::time;

// Tait EOS pressure constants
const RHO_0: f32 = 1000.0; // The rest density in kg/m^3
const TARGET_FRAME_TIME: f32 = 1.0 / 60.0;
pub const SMOOTHING_RADIUS: f32 = 10.0;

lazy_static! {
    static ref SIGMA_2: f32 = 15.0 / (7.0 * PI * SMOOTHING_RADIUS.powi(2));
}

const BOUNDARY_WIDTH: f32 = 300.0; // in Meters
const HALF_WIDTH: f32 = BOUNDARY_WIDTH / 2.0;
const BOUNDARY_HEIGHT: f32 = 150.0; // in Meters
const HALF_HEIGHT: f32 = BOUNDARY_HEIGHT / 2.0;

const GRAVITATIONAL_CONSTANT: f32 = 120.0;

pub struct Simulation {
    pub mass: Vec<f32>,
    pub position: Vec<Vector2<f32>>,
    pub half_step_velocity: Vec<Vector2<f32>>,
    pub acceleration: Vec<Vector2<f32>>,
    pub pressure_force: Vec<Vector2<f32>>,
    pub viscous_force: Vec<Vector2<f32>>,
    pub density: Vec<f32>,
    pub pressure: Vec<f32>,
    pub particles: usize,

    running: bool,
    step_once: bool,
}

impl Simulation {
    pub fn new(n: usize) -> Self {
        let mut new_self = Self {
            mass: vec![1.0; n],
            pressure: vec![0.0; n],
            density: vec![0.0; n],
            position: vec![Vector2::zero(); n],
            half_step_velocity: vec![Vector2::zero(); n],
            acceleration: vec![Vector2::zero(); n],
            pressure_force: vec![Vector2::zero(); n],
            viscous_force: vec![Vector2::zero(); n],
            particles: n,

            running: false,
            step_once: false,
        };

        //new_self.place_particles_in_uniform_grid(INITIAL_SPACING);
        new_self.place_particles_randomly_in_boundary();
        new_self.initialize_leapfrog();

        new_self
    }

    fn get_timestep(&self) -> (usize, f32) {
        let v_max = self
            .half_step_velocity
            .par_iter()
            .map(|v| v.magnitude())
            .reduce(|| 0.0, f32::max);

        dbg!(v_max);

        let c = 10.0;
        if v_max > 0.0 {
            let timestep = (c * SMOOTHING_RADIUS / v_max).min(TARGET_FRAME_TIME);
            let iterations = dbg!(TARGET_FRAME_TIME / timestep).ceil() as usize;
            (iterations, timestep)
        } else {
            (1, TARGET_FRAME_TIME)
        }
    }

    fn place_particles_in_uniform_grid(&mut self, spacing: f32) {
        let width_u: usize = (self.particles as f32).sqrt().round() as usize;
        let width_f: f32 = spacing * (width_u as f32);

        let height_u: usize = (self.particles as f32).sqrt().round() as usize;
        let height_f: f32 = spacing * (height_u as f32);

        let x_0: f32 = -width_f / 2.0;
        let y_0: f32 = height_f / 2.0;
        (0..)
            .cartesian_product(0..width_u)
            .take(self.particles)
            .enumerate()
            .for_each(|(i, (y, x))| {
                dbg!((x, y));
                self.position[i] = vec2(x_0 + spacing * (x as f32), y_0 - spacing * (y as f32));
            });
    }

    fn place_particles_randomly_in_boundary(&mut self) {
        self.position.par_iter_mut().for_each(|p| {
            let r_x = BOUNDARY_WIDTH * random::<f32>();
            let r_y = BOUNDARY_HEIGHT * random::<f32>();

            *p = vec2(r_x - HALF_WIDTH, r_y - HALF_HEIGHT)
        });
    }

    fn initialize_leapfrog(&mut self) {
        self.update_densities();
        self.update_pressures();
        self.update_accelerations();

        (0..self.particles).for_each(|i| {
            self.half_step_velocity[i] -= 0.5 * self.acceleration[i] * TARGET_FRAME_TIME
        });
    }

    // Updates the densities array
    fn update_densities(&mut self) {
        let densities: Vec<f32> = (0..self.particles)
            .into_par_iter()
            .map(|i| {
                (0..self.particles)
                    .map(|j| {
                        let r = self.position[j] - self.position[i];
                        self.mass[j] * w_spiky(r.magnitude(), SMOOTHING_RADIUS)
                    })
                    .sum::<f32>()
            })
            .collect();

        self.density.copy_from_slice(&densities);
    }

    /// Calculates the pressure for a particle using a linear relationship.
    fn linear_pressure(&self, i: usize) -> f32 {
        let pressure_multiplier: f32 = 10.0;
        pressure_multiplier * (self.density[i] - RHO_0)
    }

    fn update_pressures(&mut self) {
        let pressures: Vec<f32> = (0..self.particles)
            .into_par_iter()
            .map(|i| self.linear_pressure(i))
            .collect();

        self.pressure.copy_from_slice(&pressures);
    }

    fn get_pressure_force(&self, i: usize) -> Vector2<f32> {
        (0..self.particles)
            .into_par_iter()
            .map(|j| {
                if i == j {
                    return Vector2::zero();
                }

                let r = self.position[j] - self.position[i];
                let dist = r.magnitude();
                let r_hat = if dist != 0.0 {
                    r / dist
                } else {
                    let angle = 2.0 * PI * random::<f32>();
                    vec2(angle.cos(), angle.sin())
                };

                let slope = slope_w_spiky(dist, SMOOTHING_RADIUS);

                -r_hat * slope * self.mass[i] * (self.pressure[i] + self.pressure[j]) / (2.0 * self.density[j])
            })
            .sum()
    }

    fn get_gravitational_force() -> Vector2<f32> {
        -Vector2::unit_y() * GRAVITATIONAL_CONSTANT
    }

    /// Gets the acceleration for a given particle
    fn get_acceleration(&self, i: usize) -> Vector2<f32> {
        self.get_pressure_force(i) + Self::get_gravitational_force()
    }

    fn bounce_particles_on_walls(&mut self) {
        (0..self.particles).for_each(|i| {
            if self.position[i].x.abs() > HALF_WIDTH {
                self.half_step_velocity[i].x = -self.half_step_velocity[i].x * 0.9;
                self.position[i].x = self.position[i].x.signum() * HALF_WIDTH;
            }
            if self.position[i].y.abs() > HALF_HEIGHT {
                self.half_step_velocity[i].y = -self.half_step_velocity[i].y * 0.9;
                self.position[i].y = self.position[i].y.signum() * HALF_HEIGHT;
            }
        });
    }

    /// Updates the list of accelerations.
    fn update_accelerations(&mut self) {
        let accelerations: Vec<Vector2<f32>> = (0..self.particles)
            .into_par_iter()
            .map(|i| self.get_acceleration(i))
            .collect();
        self.acceleration.copy_from_slice(&accelerations);
    }

    /// Performs one leapfrog integration on velocity
    fn leapfrog_position(&mut self, timestep: f32) {
        self.position
            .par_iter_mut()
            .zip(&self.half_step_velocity)
            .for_each(|(pos, vel)| *pos += *vel * timestep);
    }

    /// Performs one leapfrog integration on position
    fn leapfrog_velocities(&mut self, timestep: f32) {
        self.half_step_velocity
            .par_iter_mut()
            .zip(&self.acceleration)
            .for_each(|(vel, acc)| *vel += *acc * timestep);
    }

    // Steps the simulation once
    pub fn step_simulation(&mut self) {
        if !self.running && !self.step_once {
            return;
        }
        self.step_once = false;

        let (iterations, timestep) = self.get_timestep();
        dbg!(iterations, timestep);

        for _ in 0..iterations {
            self.bounce_particles_on_walls();

            time_function("densities", || self.update_densities());
            time_function("pressures", || self.update_pressures());
            time_function("accelerations", || self.update_accelerations());

            // Integrate velocity and position
            self.leapfrog_velocities(timestep);
            self.leapfrog_position(timestep);
        }
    }

    pub fn step_simulation_once(&mut self) {
        self.step_once = true;
    }

    // Toggles pause
    pub fn toggle_pause(&mut self) {
        self.running = !self.running;
    }
}

fn w_spiky(dist: f32, radi: f32) -> f32 {
    if dist > radi {
        return 0.0;
    }

    let volume = PI * radi.powi(5) / 10.0;
    (radi - dist).powi(3) / volume
}

fn slope_w_spiky(dist: f32, radi: f32) -> f32 {
    if dist > radi {
        return 0.0;
    }

    let scale = -30.0 / (PI * radi.powi(5));
    (radi - dist).powi(2) * scale
}

fn time_function<F: FnMut()>(message: &str, mut f: F) {
    let b4 = Instant::now();

    f();

    let delay = Instant::now() - b4;
    let time_in_millis = delay.as_millis();

    println!("function <{message}> ran in {time_in_millis}ms");
}
