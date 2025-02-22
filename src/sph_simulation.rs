use std::{f32::consts::PI, time::Instant};

use cgmath::{vec2, InnerSpace, Vector2, Zero};
use itertools::Itertools;
use rand::random;

use rayon::prelude::*;

// Tait EOS pressure constants
pub const RHO_0: f32 = 100.0; // The rest density in kg/m^3
const TARGET_FRAME_TIME: f32 = 1.0 / 60.0;
pub const SMOOTHING_RADIUS: f32 = 6.0;

const BOUNDARY_WIDTH: f32 = 900.0; // in Meters
const HALF_WIDTH: f32 = BOUNDARY_WIDTH / 2.0;
const BOUNDARY_HEIGHT: f32 = 500.0; // in Meters
const HALF_HEIGHT: f32 = BOUNDARY_HEIGHT / 2.0;

const GRAVITATIONAL_CONSTANT: f32 = 9.81;

pub struct Simulation {
    pub mass: Vec<f32>,
    pub position: Vec<Vector2<f32>>,
    pub predicted_position: Vec<Vector2<f32>>,
    pub half_step_velocity: Vec<Vector2<f32>>,
    pub acceleration: Vec<Vector2<f32>>,
    pub pressure_force: Vec<Vector2<f32>>,
    pub viscous_force: Vec<Vector2<f32>>,
    pub density: Vec<f32>,
    pub pressure: Vec<f32>,

    //pub bp_particle: Vec<Vector2<f32>>,
    //pub bp_mass: Vec<Vector2<f32>>,
    pub particles: usize,
    pub spacial_hashing: SpatialHashGrid,

    running: bool,
    step_once: bool,
}

impl Simulation {
    pub fn new(n: usize) -> Self {
        let num_cells = n * 2;

        let mut new_self = Self {
            mass: vec![5.0; n],
            pressure: vec![0.0; n],
            density: vec![0.0; n],
            position: vec![Vector2::zero(); n],
            predicted_position: vec![Vector2::zero(); n],
            half_step_velocity: vec![Vector2::zero(); n],
            acceleration: vec![Vector2::zero(); n],
            pressure_force: vec![Vector2::zero(); n],
            viscous_force: vec![Vector2::zero(); n],
            particles: n,

            spacial_hashing: SpatialHashGrid::new(num_cells, n, SMOOTHING_RADIUS * 2.0),

            running: false,
            step_once: false,
        };

        //new_self.place_particles_in_uniform_grid(2.0);
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

        let c = 0.4;
        if v_max > 0.0 {
            let timestep = (c * SMOOTHING_RADIUS / v_max).min(TARGET_FRAME_TIME);
            let iterations = dbg!(TARGET_FRAME_TIME / timestep).ceil() as usize;
            (iterations, timestep)
        } else {
            (1, TARGET_FRAME_TIME)
        }
    }

    fn place_particles_randomly_in_boundary(&mut self) {
        self.position.par_iter_mut().for_each(|p| {
            let r_x = BOUNDARY_WIDTH * random::<f32>();
            let r_y = BOUNDARY_HEIGHT * random::<f32>();

            *p = vec2(r_x - HALF_WIDTH, r_y - HALF_HEIGHT)
        });

        self.predicted_position.copy_from_slice(&self.position);
        self.spacial_hashing.update_particles(&self.position);
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
                let neighbours = self
                    .spacial_hashing
                    .query_position(self.predicted_position[i], 2.0 * SMOOTHING_RADIUS);

                neighbours
                    .iter()
                    .map(|&j| {
                        let r = self.predicted_position[j] - self.predicted_position[i];
                        self.mass[j] * w_spiky(r.magnitude(), SMOOTHING_RADIUS)
                    })
                    .sum()
            })
            .collect();

        self.density.copy_from_slice(&densities);
    }

    /// Calculates the pressure for a particle using an equation of state. This is usually used for
    /// incompressible fluids
    fn eos_pressure(&self, i: usize) -> f32 {
        const GAMMA: i32 = 7;
        const B: f32 = 1E4;

        B * ((self.density[i] / RHO_0) - 1.0).powi(GAMMA)
    }

    fn update_pressures(&mut self) {
        let pressures: Vec<f32> = (0..self.particles)
            .into_par_iter()
            .map(|i| self.eos_pressure(i))
            .collect();

        self.pressure.copy_from_slice(&pressures);
    }

    fn get_interactive_forces(&self, i: usize) -> Vector2<f32> {
        const VISC: f32 = 0.10;

        let neighbours = self
            .spacial_hashing
            .query_position(self.position[i], 2.0 * SMOOTHING_RADIUS);

        neighbours
            .par_iter()
            .map(|&j| {
                if i == j {
                    return Vector2::zero();
                }

                let r = self.predicted_position[j] - self.predicted_position[i];
                let dist = r.magnitude();
                let r_hat = if dist != 0.0 {
                    r / dist
                } else {
                    let angle = 2.0 * PI * random::<f32>();
                    vec2(angle.cos(), angle.sin())
                };

                let grad = slope_w_spiky(dist, SMOOTHING_RADIUS) * r_hat;
                let lapl = laplacian_w_spiky(dist, SMOOTHING_RADIUS);

                let pressure = -grad * self.mass[j] * (self.pressure[i] + self.pressure[j])
                    / (2.0 * self.density[j]);

                let viscous = VISC
                    * self.mass[j]
                    * (self.half_step_velocity[j] - self.half_step_velocity[i])
                    * lapl
                    / self.density[j];

                pressure + viscous
            })
            .sum::<Vector2<f32>>()
    }

    fn get_gravitational_acceleration(&self) -> Vector2<f32> {
        -Vector2::unit_y() * GRAVITATIONAL_CONSTANT
    }

    /// Gets the acceleration for a given particle
    fn get_acceleration(&self, i: usize) -> Vector2<f32> {
        self.get_interactive_forces(i) / self.density[i] + self.get_gravitational_acceleration()
    }

    /// Updates the list of accelerations.
    fn update_accelerations(&mut self) {
        let accelerations: Vec<Vector2<f32>> = (0..self.particles)
            .into_par_iter()
            .map(|i| self.get_acceleration(i))
            .collect();
        self.acceleration.copy_from_slice(&accelerations);
    }

    fn bounce_particles_on_walls(&mut self) {
        (0..self.particles).for_each(|i| {
            if self.position[i].x.abs() > HALF_WIDTH {
                self.half_step_velocity[i].x = -self.half_step_velocity[i].x * 0.8;
                self.position[i].x = self.position[i].x.signum() * HALF_WIDTH;
            }
            if self.position[i].y.abs() > HALF_HEIGHT {
                self.half_step_velocity[i].y = -self.half_step_velocity[i].y * 0.8;
                self.position[i].y = self.position[i].y.signum() * HALF_HEIGHT;
            }
        });
    }

    fn update_predicted_positions(&mut self, timestep: f32) {
        let p = self
            .position
            .par_iter_mut()
            .zip(&self.half_step_velocity)
            .map(|(pos, vel)| *pos + *vel * timestep)
            .collect::<Vec<_>>();

        self.predicted_position.copy_from_slice(&p);
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
            self.update_predicted_positions(timestep);

            self.spacial_hashing
                .update_particles(&self.predicted_position);

            time_function("densities", || self.update_densities());
            time_function("pressures", || self.update_pressures());
            time_function("accelerations", || self.update_accelerations());

            // Integrate velocity and position
            self.leapfrog_velocities(timestep);
            self.leapfrog_position(timestep);

            self.bounce_particles_on_walls();
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

fn laplacian_w_spiky(dist: f32, radi: f32) -> f32 {
    if dist > radi {
        return 0.0;
    }

    let scale = 60.0 / (PI * radi.powi(5));
    (radi - dist) * scale
}

fn time_function<F: FnMut()>(message: &str, mut f: F) {
    let b4 = Instant::now();

    f();

    let delay = Instant::now() - b4;
    let time_in_millis = delay.as_millis();

    println!("function <{message}> ran in {time_in_millis}ms");
}

pub struct SpatialHashGrid {
    pub spacing: f32,
    num_cells: usize,
    cell_count: Vec<usize>,
    particles_map: Vec<usize>,
}

impl SpatialHashGrid {
    fn new(num_cells: usize, num_particles: usize, spacing: f32) -> Self {
        Self {
            spacing,
            num_cells,
            cell_count: vec![0; num_cells + 1],
            particles_map: vec![0; num_particles],
        }
    }

    /// Hash function. Takes a 2d grid cell and maps it to a 1d array.
    fn hash_coords(&self, xi: isize, yi: isize) -> usize {
        (((xi * 92837111) ^ (yi * 689287499)).rem_euclid(self.num_cells as isize)) as usize
    }

    /// Takes a position in 2d space and maps it to a grid cell.
    pub fn int_coords(&self, location: Vector2<f32>) -> Vector2<isize> {
        let (x, y) = (location.x, location.y);
        vec2(
            (x / self.spacing).floor() as isize,
            (y / self.spacing).floor() as isize,
        )
    }

    /// Takes a position and maps it to an array slot.
    fn hash_pos(&self, location: Vector2<f32>) -> usize {
        let Vector2 { x, y } = self.int_coords(location);
        self.hash_coords(x, y)
    }

    /// Inserts all particles into the particles map.
    fn update_particles(&mut self, positions: &[Vector2<f32>]) {
        self.cell_count.fill(0);
        self.particles_map.fill(0);

        // Count number of particles in each cell
        positions.iter().for_each(|p| {
            let i = self.hash_pos(*p);
            self.cell_count[i] += 1;
        });

        // Populate all grid cells
        let mut start = 0;
        (0..self.num_cells).for_each(|i| {
            start += self.cell_count[i];
            self.cell_count[i] = start;
        });
        self.cell_count[self.num_cells] = start;

        // Insert particles into particles_map
        positions.iter().enumerate().for_each(|(i, p)| {
            let hash = self.hash_pos(*p);
            // Decrement counter to give each particle a unique position
            self.cell_count[hash] -= 1;

            let particle_pos = self.cell_count[hash];
            self.particles_map[particle_pos] = i;
        });
    }

    /// Returns the particles within the given radius at the given position
    pub fn query_position(&self, position: Vector2<f32>, radius: f32) -> Vec<usize> {
        let r = vec2(radius, radius);
        let Vector2 { x: x0, y: y0 } = self.int_coords(position - r);
        let Vector2 { x: x1, y: y1 } = self.int_coords(position + r);

        let mut particles = Vec::new();

        (x0..=x1).cartesian_product(y0..=y1).for_each(|(xi, yi)| {
            let h = self.hash_coords(xi, yi);
            let start = self.cell_count[h];
            let end = self.cell_count[h + 1];

            for i in start..end {
                let particle = self.particles_map[i];
                particles.push(particle);
            }
        });

        particles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Spatial Hash Grid Tests
    #[test]
    fn insert_particles() {
        let (particles, mut grid) = grid_scenario_1();

        grid.update_particles(&particles);
    }

    fn grid_scenario_1() -> (Vec<Vector2<f32>>, SpatialHashGrid) {
        let grid_spacing = 1.0;
        let particles = vec![
            vec2(0.5, 0.2), // 0,0 : 0
            vec2(0.7, 0.7), // 0,0 : 1
            vec2(2.1, 1.4), // 2,1 : 2
            vec2(2.1, 1.8), // 2,1 : 3
            vec2(1.8, 2.3), // 1,2 : 4
        ];

        let n_particles = particles.len();
        let grid = SpatialHashGrid::new(2 * n_particles, n_particles, grid_spacing);

        (particles, grid)
    }
}
