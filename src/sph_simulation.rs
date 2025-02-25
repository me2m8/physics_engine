use std::{collections::HashSet, f32::consts::PI, process::exit, time::Instant};

use cgmath::{vec2, InnerSpace, Matrix2, MetricSpace, Vector2, Zero};
use itertools::Itertools;
use rand::random;

use rayon::prelude::*;

// Tait EOS pressure constants
pub const RHO_0: f32 = 15.0; // The rest density in kg/m^3
pub const RHO_MAX: f32 = RHO_0 * 1.5; // The rest density in kg/m^3
const TARGET_FRAME_TIME: f32 = 1.0 / 60.0;
pub const SMOOTHING_RADIUS: f32 = 15.0;

pub const BOUNDARY_WIDTH: f32 = 1000.0; // in Meters
pub const HALF_WIDTH: f32 = BOUNDARY_WIDTH / 2.0;
pub const BOUNDARY_HEIGHT: f32 = 650.0; // in Meters
pub const HALF_HEIGHT: f32 = BOUNDARY_HEIGHT / 2.0;

const GRAVITATIONAL_CONSTANT: f32 = 145.81;

pub struct Simulation {
    pub mass: Vec<f32>,
    pub position: Vec<Vector2<f32>>,
    pub predicted_position: Vec<Vector2<f32>>,
    pub half_step_velocity: Vec<Vector2<f32>>,
    pub acceleration: Vec<Vector2<f32>>,
    pub pressure_force: Vec<Vector2<f32>>,
    pub viscous_force: Vec<Vector2<f32>>,
    pub adhesion_force: Vec<Vector2<f32>>,
    pub surface_tension_force: Vec<Vector2<f32>>,
    pub density: Vec<f32>,
    pub pressure: Vec<f32>,
    pub color_field: Vec<Vector2<f32>>,

    pub boundary_particle_position: Vec<Vector2<f32>>,
    pub boundary_particles: usize,

    //pub bp_particle: Vec<Vector2<f32>>,
    //pub bp_mass: Vec<Vector2<f32>>,
    pub particles: usize,
    pub spacial_hashing: SpatialHashGrid,
    pub boundary_spacial_hashing: SpatialHashGrid,

    running: bool,
    step_once: bool,
}

impl Simulation {
    pub fn new(n: usize) -> Self {
        let mut new_self = Self {
            mass: vec![1000.0; n],
            pressure: vec![0.0; n],
            density: vec![0.0; n],
            position: vec![Vector2::zero(); n],
            predicted_position: vec![Vector2::zero(); n],
            half_step_velocity: vec![Vector2::zero(); n],

            acceleration: vec![Vector2::zero(); n],
            pressure_force: vec![Vector2::zero(); n],
            viscous_force: vec![Vector2::zero(); n],
            adhesion_force: vec![Vector2::zero(); n],
            surface_tension_force: vec![Vector2::zero(); n],
            color_field: vec![Vector2::zero(); n],

            particles: n,

            boundary_particle_position: Default::default(),
            boundary_particles: 0,

            spacial_hashing: SpatialHashGrid::new(n, SMOOTHING_RADIUS * 2.0),
            boundary_spacial_hashing: SpatialHashGrid::new(0, SMOOTHING_RADIUS * 2.0),

            running: false,
            step_once: false,
        };

        //new_self.place_particles_randomly_in_boundary();
        new_self.place_particles_uniformally(None);
        new_self.place_boundary_particles(5.0);
        new_self.initialize_leapfrog();

        new_self
    }

    pub fn add_new_particle(&mut self, position: Vector2<f32>) {
        let o = Vector2::zero();
        self.position.push(position);
        self.mass.push(1000.0);
        self.pressure.push(0.0);
        self.density.push(0.0);
        self.predicted_position.push(position);
        self.half_step_velocity.push(o);
        self.acceleration.push(o);
        self.pressure_force.push(o);
        self.viscous_force.push(o);
        self.surface_tension_force.push(o);
        self.color_field.push(o);

        self.particles += 1;

        self.spacial_hashing.grow(self.particles);
    }

    pub fn add_many_new_particles(&mut self, position: Vector2<f32>, spacing: f32) {
        (-4..=4).cartesian_product(-4..=4).for_each(|(i, j)| {
            let p = position + vec2(i as f32 * spacing, j as f32 * spacing);
            self.add_new_particle(p);
        });
    }

    fn get_timestep(&self) -> (usize, f32) {
        let v_max = self
            .half_step_velocity
            .par_iter()
            .map(|v| v.magnitude())
            .reduce(|| 0.0, f32::max);

        let c = 0.4;
        if v_max > 0.0 {
            let timestep = (c * SMOOTHING_RADIUS / v_max).min(TARGET_FRAME_TIME);
            let iterations = (TARGET_FRAME_TIME / timestep).ceil() as usize;
            (iterations, timestep)
        } else {
            (1, TARGET_FRAME_TIME)
        }
    }

    fn place_boundary_particles(&mut self, spacing: f32) {
        let mut count = 0;

        let nh = (BOUNDARY_WIDTH / spacing).round() as usize;
        let nv = (BOUNDARY_HEIGHT / spacing).round() as usize;

        let thickness = (SMOOTHING_RADIUS / spacing) as usize + 1;

        let y_0 = HALF_HEIGHT + (thickness as f32) * spacing;
        let x_0 = -HALF_WIDTH - (thickness as f32) * spacing;

        (0..=nv + 2 * thickness)
            .cartesian_product(0..thickness)
            .for_each(|(y_i, t_i)| {
                count += 2;

                let x = x_0 + (t_i as f32) * spacing;
                let y = y_0 - (y_i as f32) * spacing;

                self.boundary_particle_position.push(vec2(x, y));
                self.boundary_particle_position.push(vec2(-x, y));
            });

        (0..=nh)
            .cartesian_product(0..thickness)
            .for_each(|(x_i, t_i)| {
                count += 2;

                let y = y_0 - (t_i as f32) * spacing;
                let x = x_0 + ((thickness + x_i) as f32) * spacing;

                self.boundary_particle_position.push(vec2(x, y));
                self.boundary_particle_position.push(vec2(x, -y));
            });

        self.boundary_particles = count;
        self.boundary_spacial_hashing = SpatialHashGrid::new(count, SMOOTHING_RADIUS * 2.0);
        self.boundary_spacial_hashing
            .update_particles(&self.boundary_particle_position);
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

    fn place_particles_uniformally(&mut self, spacing: Option<f32>) {
        let ratio = BOUNDARY_WIDTH / BOUNDARY_HEIGHT;
        let x_width = (self.particles as f32 * ratio).sqrt().floor() as usize;
        let spacing = spacing.unwrap_or((BOUNDARY_WIDTH - 50.0) / x_width as f32);

        let x_0 = -(BOUNDARY_WIDTH - 10.0) / 2.0;
        let y_0 = -(BOUNDARY_HEIGHT - 10.0) / 2.0;

        let positions = (0..)
            .cartesian_product(0..x_width)
            .map(|(y, x)| {
                let x = x_0 + x as f32 * spacing;
                let y = y_0 + y as f32 * spacing;

                vec2(x, y)
            })
            .take(self.particles)
            .collect::<Vec<_>>();

        self.position.copy_from_slice(&positions);
        self.predicted_position.copy_from_slice(&positions);
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
                let f_neighbours = self
                    .spacial_hashing
                    .query_position(self.predicted_position[i], SMOOTHING_RADIUS);

                let b_neighbours = self
                    .boundary_spacial_hashing
                    .query_position(self.predicted_position[i], SMOOTHING_RADIUS);

                let f_density: f32 = f_neighbours
                    .par_iter()
                    .map(|&j| {
                        let r = self.predicted_position[j] - self.predicted_position[i];
                        self.mass[j] * w_spiky(r.magnitude(), SMOOTHING_RADIUS)
                    })
                    .sum();

                let b_density: f32 = b_neighbours
                    .par_iter()
                    .map(|&j| {
                        let r = self.boundary_particle_position[j] - self.predicted_position[i];
                        self.mass[i] * w_spiky(r.magnitude(), SMOOTHING_RADIUS)
                    })
                    .sum();

                f_density + b_density
            })
            .collect();

        self.density.copy_from_slice(&densities);
    }

    fn diffuse_densities(&mut self) {
        const LAMBDA: f32 = 0.01;

        let densities = (0..self.particles)
            .into_par_iter()
            .map(|i| {
                let xi = self.predicted_position[i];

                let neighbors = self.spacial_hashing.query_position(xi, SMOOTHING_RADIUS);

                self.density[i]
                    + LAMBDA
                        * neighbors
                            .par_iter()
                            .map(|&j| {
                                let xj = self.predicted_position[j];
                                let r = (xi - xj).magnitude();

                                self.mass[j]
                                    * (self.density[i] / self.density[j] - 1.0)
                                    * w_spiky(r, SMOOTHING_RADIUS)
                            })
                            .sum::<f32>()
            })
            .collect::<Vec<_>>();

        self.density.copy_from_slice(&densities);
    }

    fn linear_pressure(&self, i: usize) -> f32 {
        const K: f32 = 3E6;

        K * (self.density[i] - RHO_0)
    }

    /// Calculates the pressure for a particle using an equation of state. This is usually used for
    /// incompressible fluids
    fn eos_pressure(&self, i: usize) -> f32 {
        const GAMMA: i32 = 7;
        const B: f32 = 1E6;

        B * ((self.density[i] / RHO_0).powi(GAMMA) - 1.0)
    }

    fn update_pressures(&mut self) {
        let pressures: Vec<f32> = (0..self.particles)
            .into_par_iter()
            .map(|i| self.linear_pressure(i))
            .collect();

        self.pressure.copy_from_slice(&pressures);
    }

    fn update_interactive_forces(&mut self) {
        const F_VISC: f32 = 300.00;
        const B_VISC: f32 = 1000.00;

        let (pf, vf): (Vec<_>, Vec<_>) = (0..self.particles)
            .into_par_iter()
            .map(|i| {
                let f_neighbours = self
                    .spacial_hashing
                    .query_position(self.predicted_position[i], SMOOTHING_RADIUS);

                let b_neighbours = self
                    .boundary_spacial_hashing
                    .query_position(self.predicted_position[i], SMOOTHING_RADIUS);

                let f_forces = f_neighbours
                    .par_iter()
                    .map(|&j| {
                        if i == j {
                            return Matrix2::zero();
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
                        let lapl = viscosity_kernel(dist, SMOOTHING_RADIUS);

                        // Symmetric pressure force
                        let pressure = grad
                            * self.mass[j]
                            * (self.pressure[i] / self.density[i].powi(2)
                                + self.pressure[j] / self.density[j].powi(2));

                        // Viscous force gotten directly from the navier stokes equations
                        let viscous = F_VISC
                            * self.mass[j]
                            * (self.half_step_velocity[j] - self.half_step_velocity[i])
                            * lapl
                            / self.density[j];

                        // Viscous force from finite difference approximation
                        //let viscous = VISC * 8.0 * self.mass[j] / self.density[j]
                        //    * (self.half_step_velocity[i] - self.half_step_velocity[j]).dot(r)
                        //    / (dist_2 + 0.01 * SMOOTHING_RADIUS.powi(2))
                        //    * grad;

                        Matrix2::from_cols(pressure, viscous)
                    })
                    .sum::<Matrix2<f32>>();

                // Calculate forces from boundary particles
                let b_forces = b_neighbours
                    .par_iter()
                    .map(|&j| {
                        let r = self.boundary_particle_position[j] - self.predicted_position[i];
                        let dist = r.magnitude();
                        let r_hat = if dist != 0.0 {
                            r / dist
                        } else {
                            let angle = 2.0 * PI * random::<f32>();
                            vec2(angle.cos(), angle.sin())
                        };

                        let grad = slope_w_spiky(dist, SMOOTHING_RADIUS) * r_hat;
                        let lapl = viscosity_kernel(dist, SMOOTHING_RADIUS);

                        // Symmetric pressure force
                        let pressure = grad
                            * (self.mass[i]
                                * (self.pressure[i] / self.density[i].powi(2)
                                    + self.pressure[i] / RHO_0.powi(2)))
                            .max(0.0);

                        // Viscous force gotten directly from the navier stokes equations
                        let viscous =
                            B_VISC * self.mass[i] * (-self.half_step_velocity[i]) * lapl / RHO_0;

                        Matrix2::from_cols(pressure, viscous)
                    })
                    .sum::<Matrix2<f32>>();

                let forces = f_forces + b_forces;

                (forces.x, forces.y)
            })
            .unzip();

        self.pressure_force.copy_from_slice(&pf);
        self.viscous_force.copy_from_slice(&vf);
    }

    fn update_color_field(&mut self) {
        let h = 1.0;

        let color_field = (0..self.particles)
            .into_par_iter()
            .map(|i| {
                let xi = self.predicted_position[i];
                let f_neighbors = self.spacial_hashing.query_position(xi, SMOOTHING_RADIUS);
                let b_neighbors = self
                    .boundary_spacial_hashing
                    .query_position(xi, SMOOTHING_RADIUS);

                h * (f_neighbors
                    .par_iter()
                    .map(|&j| {
                        if i == j {
                            return Vector2::zero();
                        }

                        let xj = self.predicted_position[j];
                        let r = xi - xj;

                        let volume = self.mass[j] / self.density[j];
                        let slope = slope_w_spiky(r.magnitude(), SMOOTHING_RADIUS);
                        let dir = r.normalize();

                        volume * slope * dir
                    })
                    .sum::<Vector2<f32>>()
                    + b_neighbors
                        .par_iter()
                        .map(|&j| {
                            let xj = self.boundary_particle_position[j];
                            let r = xi - xj;

                            let volume = self.mass[i] / RHO_0;
                            let slope = slope_w_spiky(r.magnitude(), SMOOTHING_RADIUS);
                            let dir = r.normalize();

                            volume * slope * dir
                        })
                        .sum::<Vector2<f32>>())
            })
            .collect::<Vec<_>>();

        self.color_field.copy_from_slice(&color_field);
    }

    fn update_surface_tension_forces(&mut self) {
        const ALPHA: f32 = 2.50;
        let cohesion: Vec<_> = (0..self.particles)
            .into_par_iter()
            .map(|i| {
                let xi = self.predicted_position[i];
                let neighbours = self.spacial_hashing.query_position(xi, SMOOTHING_RADIUS);

                // #! Macroscopic approach to surface tension
                -ALPHA
                    * neighbours
                        .par_iter()
                        .map(|&j| {
                            if i == j {
                                return Vector2::zero();
                            }

                            let xj = self.predicted_position[j];
                            let r = xi - xj;

                            let cohesion = self.mass[i]
                                * self.mass[j]
                                * r.normalize()
                                * cohesion_kernel(r.magnitude(), SMOOTHING_RADIUS);

                            let curvature =
                                self.mass[i] * (self.color_field[i] - self.color_field[j]);

                            let k = RHO_0 * 2.0 / (self.density[i] + self.density[j]);

                            k * cohesion + curvature
                        })
                        .sum::<Vector2<f32>>()
            })
            .collect();

        self.surface_tension_force.copy_from_slice(&cohesion);
    }

    fn update_adhesive_forces(&mut self) {
        const BETA: f32 = 1.0;
    }

    fn get_gravitational_acceleration(&self) -> Vector2<f32> {
        -Vector2::unit_y() * GRAVITATIONAL_CONSTANT
    }

    /// Updates the list of accelerations.
    fn update_accelerations(&mut self) {
        self.update_interactive_forces();
        let accelerations: Vec<Vector2<f32>> = (0..self.particles)
            .into_par_iter()
            .map(|i| {
                (self.pressure_force[i] + self.viscous_force[i] + self.surface_tension_force[i])
                    / self.density[i]
                    + self.get_gravitational_acceleration()
            })
            .collect();
        self.acceleration.copy_from_slice(&accelerations);
    }

    fn bounce_particles_on_walls(&mut self) {
        (0..self.particles).for_each(|i| {
            if self.position[i].x.abs() > HALF_WIDTH {
                self.half_step_velocity[i].x = -self.half_step_velocity[i].x * 0.50;
                self.position[i].x = self.position[i].x.signum() * HALF_WIDTH;
            }
            if self.position[i].y.abs() > HALF_HEIGHT {
                self.half_step_velocity[i].y = -self.half_step_velocity[i].y * 0.50;
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

        for _ in 0..iterations {
            self.update_predicted_positions(timestep);

            self.spacial_hashing
                .update_particles(&self.predicted_position);

            //time_function("densities", || self.update_densities());
            //time_function("pressures", || self.update_pressures());
            //time_function("accelerations", || self.update_accelerations());
            self.update_densities();
            self.diffuse_densities();
            self.update_pressures();
            self.update_color_field();
            self.update_surface_tension_forces();
            self.update_accelerations();

            // Integrate velocity and position
            self.leapfrog_velocities(timestep);
            self.leapfrog_position(timestep);
        }

        //while ((Instant::now() - now).as_secs_f32() < 1.0 / 60.0) {}
    }

    pub fn step_simulation_once(&mut self) {
        self.step_once = true;
    }

    // Toggles pause
    pub fn toggle_pause(&mut self) {
        self.running = !self.running;
    }

    pub fn get_closest_particle(&self, pos: Vector2<f32>) -> usize {
        let mut min = 0;
        let mut closest = f32::MAX;
        (0..self.particles).for_each(|i| {
            let dist = self.position[i].distance(pos);
            if dist < closest {
                closest = dist;
                min = i;
            }
        });

        let int_coords = self.spacial_hashing.int_coords(self.position[min]);
        println!("Int Coords: {int_coords:#?}");

        min
    }
}

pub fn w_spiky(dist: f32, radi: f32) -> f32 {
    if dist > radi {
        return 0.0;
    }

    let volume = PI * radi.powi(4) / 6.0;
    (radi - dist).powi(2) / volume
}

fn slope_w_spiky(dist: f32, radi: f32) -> f32 {
    if dist > radi {
        return 0.0;
    }

    let scale = -12.0 / (PI * radi.powi(4));
    (radi - dist) * scale
}

fn cohesion_kernel(dist: f32, radi: f32) -> f32 {
    let h = radi;
    32.0 / (PI * h.powi(9))
        * match dist {
            r if r < h / 2.0 => 2.0 * (h - r).powi(3) * r.powi(3) - h.powi(6) / 64.0,
            r if r < h => (h - r).powi(3) * r.powi(3),
            _ => 0.0,
        }
}

fn adhesion_kernel(dist: f32, radi: f32) -> f32 {
    let r = dist;
    let h = radi;

    if r > h || r <= h / 2.0 {
        return 0.0;
    }

    0.007 / h.powf(3.25) * (-4.0 * r.powi(2) / h + 6.0 * r - 2.0 * h).powf(0.25)
}

fn viscosity_kernel(dist: f32, radi: f32) -> f32 {
    if dist > radi {
        return 0.0;
    }

    let q = dist / radi;

    let scale = 40.0 / (PI * radi.powi(4));
    (1.0 - q).powi(3) * scale
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
    particles_map: Vec<usize>,
    cell_count: Vec<usize>,
}

impl SpatialHashGrid {
    fn new(num_particles: usize, spacing: f32) -> Self {
        Self {
            spacing,

            num_cells: 2 * num_particles,
            cell_count: vec![0; 2 * num_particles + 1],
            particles_map: vec![0; num_particles],
        }
    }

    fn grow(&mut self, new_size: usize) {
        self.num_cells = 2 * new_size;
        self.cell_count.resize(self.num_cells + 1, 0);
        self.particles_map.resize(new_size, 0);
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

        let mut particles = HashSet::new();

        (x0..=x1).cartesian_product(y0..=y1).for_each(|(xi, yi)| {
            let h = self.hash_coords(xi, yi);
            let start = self.cell_count[h];
            let end = self.cell_count[h + 1];

            for i in start..end {
                let particle = self.particles_map[i];
                particles.insert(particle);
            }
        });

        particles.into_iter().collect_vec()
    }
}
