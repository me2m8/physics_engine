use std::f32::consts::PI;

use cgmath::{
    num_traits::{real::Real, zero, Float},
    vec2, Deg, InnerSpace, MetricSpace, Vector2, Zero,
};
use itertools::Itertools;
use rand::random;

const SLOWDOWN: f32 = 1.0;

const TIMESTEP: f32 = 1.0 / (60.0 * SLOWDOWN);

const SMOOTHING_RADIUS: f32 = 10.0;
const G: f32 = 6.674E-11;
const EPSILON: f32 = 0.1; // added when dividing by dividing distance to avoid dividing by zero

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
        };

        new_self.randomize_positions(50.0);
        new_self.initialize_leapfrog();

        new_self
    }

    fn randomize_positions(&mut self, radius: f32) {
        (0..self.particles).for_each(|i| {
            let theta: f32 = 2.0 * PI * random::<f32>();
            let r: f32 = radius * random::<f32>();
            let (sin, cos) = theta.sin_cos();

            self.position[i] = r * vec2(sin, cos);
        });
    }

    fn initialize_leapfrog(&mut self) {
        self.update_densities();
        self.update_pressures();
        self.update_accelerations();

        (0..self.particles)
            .for_each(|i| self.half_step_velocity[i] -= 0.5 * self.acceleration[i] * TIMESTEP);
    }

    fn update_densities(&mut self) {
        // Null all densities
        self.density.fill(0.0);

        // Sum up the new densities of the particles
        (0..self.particles)
            .combinations_with_replacement(2)
            .for_each(|pair| {
                let (i, j) = (pair[0], pair[1]);
                let r = self.position[j] - self.position[i];
                self.density[i] += self.mass[j] * w(r, SMOOTHING_RADIUS);
                self.density[j] += self.mass[i] * w(r, SMOOTHING_RADIUS);
            });
    }

    /// Calculates the pressure for a particle using a linear relationship.
    fn linear_pressure(&self, i: usize) -> f32 {
        const C: f32 = 1500.0; // speed of sound in m/s
        const C2: f32 = C * C;
        const REF_DENSITY: f32 = 998.0; // Density of water in kg/m^3

        C2 * (self.density[i] - REF_DENSITY)
    }

    /// Calculates the pressure for a particle using a non-linear relatoinship.
    fn tait_eos_pressure(&self, i: usize) -> f32 {
        const B: f32 = 1.0E1; // Common value for stiffness parameter
        const RHO_0: f32 = 10.0; // kg/m^3
        const GAMMA: f32 = 7.0; // Common value for water

        B * ((self.density[i] / RHO_0).powf(GAMMA) - 1.0)
    }

    fn update_pressures(&mut self) {
        (0..self.particles).for_each(|i| self.pressure[i] = self.tait_eos_pressure(i))
    }

    fn get_pressure_force(&self, i: usize) -> Vector2<f32> {
        let mut acceleration: Vector2<f32> = Vector2::zero();

        (0..self.particles).for_each(|j| {
            let r = self.position[j] - self.position[i];

            acceleration -= (self.mass[j] / self.density[j])
                * (self.pressure[i] / self.density[i].powi(2)
                    + self.pressure[j] / self.density[j].powi(2))
                * w_gradient(r, SMOOTHING_RADIUS);
        });

        acceleration
    }

    fn get_attractive_force(&self, i: usize) -> Vector2<f32> {
        // Epsilon softening is used to mitegate the effects of the gravitational force blowing up
        // for very small distances
        const EPSILON: f32 = 0.1;
        const EPSILON_2: f32 = EPSILON * EPSILON;

        let mut acceleration = Vector2::zero();

        (0..self.particles).for_each(|j| {
            if i == j {
                return;
            }

            let r = self.position[j] - self.position[i];
            acceleration += G * self.mass[j] * r.normalize() / (r.magnitude2() + EPSILON);
        });

        acceleration
    }

    //fn get_gravitational_force(&self, i: usize) -> Vector2<f32> {
    //
    //}

    /// Gets the force on a particle from the boundary. This force is proportional to the square of
    /// the distance outside the wall on each axis.
    fn get_boundary_force(&self, i: usize) -> Vector2<f32> {
        const BOUNDARY_WIDTH: f32 = 100.0; // in Meters
        const HALF_WIDTH: f32 = BOUNDARY_WIDTH / 2.0;
        const BOUNDARY_HEIGHT: f32 = 100.0; // in Meters
        const HALF_HEIGHT: f32 = BOUNDARY_HEIGHT / 2.0;
        const WALL_PRESSURE_K: f32 = 1E2; // coeff for calculating pressure from wall

        let pos = self.position[i];

        // the penetration into the wall on each axis
        // (q is a placeholder coordinate) pos.q.abs() - BOUNDARY will be negative if within the boundary, so pen.q will be 0 if within the boundary
        let (x_pen, y_pen) = (
            (pos.x.abs() - HALF_WIDTH).max(0.0),
            (pos.y.abs() - HALF_HEIGHT).max(0.0),
        );

        let x_acc = if x_pen == 0.0 {
            0.0
        } else {
            -WALL_PRESSURE_K * pos.x.signum() * x_pen * x_pen
        };

        let y_acc = if y_pen == 0.0 {
            0.0
        } else {
            -WALL_PRESSURE_K * pos.y.signum() * y_pen * y_pen
        };

        vec2(x_acc, y_acc)
    }

    /// Gets the acceleration for a given particle
    fn get_acceleration(&self, i: usize) -> Vector2<f32> {
        self.get_pressure_force(i) + self.get_boundary_force(i)
    }

    /// Updates the list of accelerations.
    fn update_accelerations(&mut self) {
        (0..self.particles).for_each(|i| {
            self.acceleration[i] = self.get_acceleration(i);
        });
    }

    /// Performs one leapfrog integration on velocity
    fn leapfrog_position(&mut self) {
        (0..self.particles).for_each(|i| self.position[i] += self.half_step_velocity[i] * TIMESTEP);
    }

    /// Performs one leapfrog integration on position
    fn leapfrog_velocities(&mut self) {
        (0..self.particles)
            .for_each(|i| self.half_step_velocity[i] += self.acceleration[i] * TIMESTEP);
    }

    pub fn update(&mut self) {
        // Update each particles state
        self.update_densities();
        self.update_pressures();
        self.update_accelerations();

        // Integrate velocity and position
        self.leapfrog_velocities();
        self.leapfrog_position();
    }
}

/// The smoothing kernel. Denoted W(r, h)
fn w(r: Vector2<f32>, h: f32) -> f32 {
    let dist = r / h;
    let mag_2 = dist.magnitude2();

    (-mag_2).exp() / (PI * h * h)
}

/// The gradient of the smoothing kernel, denoted ∇W(r, h)
fn w_gradient(r: Vector2<f32>, h: f32) -> Vector2<f32> {
    -(2.0 * r / (h * h)) * w(r, h)
}

/// The laplacian of the smoothing kernel, denoted ∇
fn w_laplacian(r: Vector2<f32>, h: f32) -> f32 {
    (4.0 * r.magnitude2() / (h.powi(4)) - 4.0 / (h * h)) * w(r, h)
}
