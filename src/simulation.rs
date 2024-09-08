use std::ops::Mul;

use cgmath::{num_traits::Float, vec2, vec4, Vector2};
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
        let br = vec4(rad, -rad, 0.0, 0.0);
        let tr = vec4(rad, rad, 0.0, 0.0);
        let tl = vec4(-rad, rad, 0.0, 0.0);

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
    u: [[f32; GRID_HEIGHT]; GRID_X_BOUNDS],
    v: [[f32; GRID_Y_BOUNDS]; GRID_WIDTH],
    div: [[f32; GRID_HEIGHT]; GRID_WIDTH],
    obst: [[u8; OBST_GRID_HEIGHT]; OBST_GRID_WIDTH],
}

const BOUNDARY_WIDTH: f32 = 1900.0;
const BOUNDARY_HEIGHT: f32 = 1060.0;

/// The amount of fluid grid columns
const GRID_WIDTH: usize = 190;
/// The amount of fluid grid rows
const GRID_HEIGHT: usize = 106;

/// The size of a grid_square in pixels
const GRID_SQUARE: f32 = 10.0;

/// The amount of obstacle grid columns, 2 more than the fluid grid to account for the boundaries
const OBST_GRID_WIDTH: usize = GRID_WIDTH + 2;
/// The amount of obstacle grid rows, 2 more than the fluid grid to account for the boundaries
const OBST_GRID_HEIGHT: usize = GRID_HEIGHT + 2;

/// The amount of x boundaries in the fluid grid
const GRID_X_BOUNDS: usize = GRID_WIDTH + 1;
/// The amount of y boundaries in the fluid grid
const GRID_Y_BOUNDS: usize = GRID_HEIGHT + 1;

/// `1 < BIG_O < 2`
const OVERRELAXATION: f32 = 1.9;
const GRAVITY: f32 = 9.81 * METERS_PER_SECOND;

const TIMESTEPS_PER_SECOND: f32 = 60.0;
/// time for a single timestep
const TIMESTEP: f32 = 1.0 / TIMESTEPS_PER_SECOND;

/// A unit for converting meters to pixels
const PIXELS_PER_METER: f32 = 1920.0;
/// A unit for converting meters per second to pixels per timestep
const METERS_PER_SECOND: f32 = PIXELS_PER_METER * TIMESTEP;

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

        particles.push(Particle {
            position: vec2(0.0, 0.0),
            radius: 10.0,
        });

        let u = [[0.0; GRID_HEIGHT]; GRID_X_BOUNDS];
        let v = [[0.0; GRID_Y_BOUNDS]; GRID_WIDTH];
        let div = [[0.0; GRID_HEIGHT]; GRID_WIDTH];
        let obst = [[1; GRID_HEIGHT + 2]; GRID_WIDTH + 2];

        Self {
            particles,
            u,
            v,
            div,
            obst,
        }
    }

    pub fn particles_to_circle_vertices(&self) -> Vec<QuadVertex> {
        self.particles
            .iter()
            .map(|p| p.to_vertices())
            .collect_vec()
            .concat()
    }

    fn update_velocities(&mut self) {
        (0..GRID_X_BOUNDS)
            .cartesian_product(0..GRID_HEIGHT)
            .for_each(|(i, j)| self.v[i][j] += GRAVITY);
    }

    fn calculate_divergence(&mut self, i: usize, j: usize) -> f32 {
        OVERRELAXATION * (self.u[i + 1][j] - self.u[i][j] + self.v[i][j + 1] - self.v[i][j])
    }

    fn solve_incompressability(&mut self) {
        (0..GRID_WIDTH)
            .cartesian_product(0..GRID_HEIGHT)
            .for_each(|(i, j)| {
                let si = i + 1;
                let sj = j + 1;

                let s = self.obst[si - 1][sj]
                    + self.obst[si + 1][sj]
                    + self.obst[si][sj - 1]
                    + self.obst[si][sj + 1];
                let d = self.calculate_divergence(i, j);

                self.u[i][j] += d * (self.obst[si - 1][sj] as f32 / s as f32);
                self.u[i + 1][j] -= d * (self.obst[si + 1][sj] as f32 / s as f32);
                self.v[i][j] += d * (self.obst[si][sj - 1] as f32 / s as f32);
                self.v[i][j + 1] -= d * (self.obst[si][sj + 1] as f32 / s as f32);
            })
    }

    fn sample_velocity_field(&mut self, position: Vector2<f32>) -> f32 {
        // Get grid coordinates
        let grid_x = position.x / GRID_SQUARE;
        let grid_y = position.y / GRID_SQUARE;

        // Get the cell coordinates
        let w_01 = grid_x.fract();
        let w_11 = grid_y.fract();
        let w_00 = 1.0 - w_01;
        let w_10 = 1.0 - w_11;

        let mut v: f32 = 0.0;
        let mut components = 0.0;

        let tj = grid_y as usize;
        let bj = grid_y as usize + 1;
        let li: usize;
        let ri: usize;

        if grid_x > 0.5 {
            li = (grid_x - 0.5).floor() as usize;
            ri = li + 1;
            v += self.v[li][tj] * w_01 * w_11;
            v += self.v[li][bj] * w_00 * w_10;
            components += 2.0;
        } else {
            ri = 0;
        }

        if grid_x < GRID_WIDTH as f32 - 0.5 {
            v += self.v[ri][tj] * w_00 * w_11;
            v += self.v[ri][bj] * w_01 * w_10;
            components += 2.0;
        }

        v /= components;

        v
    }

    fn get_vel_at_u(&mut self, ux: usize, uy: usize) -> Vector2<f32> {
        let hv = self.u[ux][uy];
        let mut vv: f32 = 0.0;
        let mut components = 2.0;

        if ux != 0 {
            if uy != GRID_HEIGHT {
                vv += self.v[ux - 1][uy + 1];
                components += 1.0;
            }
            vv += self.v[ux - 1][uy];
            components += 1.0;
        }

        if uy != GRID_HEIGHT {
            vv += self.v[ux][uy + 1];
            components += 1.0;
        }
        vv += self.v[ux][uy];
        components += 1.0;

        vv /= components;

        vec2(hv, vv)
    }

    fn advection(&mut self) {
        (0..GRID_X_BOUNDS)
            .cartesian_product(0..GRID_Y_BOUNDS)
            .for_each(|(i, j)| {

            });
    }

    pub fn tick(&mut self) {
        self.update_velocities();
        (0..5).for_each(|_| {
            self.solve_incompressability();
        });
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
    let br = vec4(hw, -hh, 0.0, 1.0);
    let tr = vec4(hw, hh, 0.0, 1.0);
    let tl = vec4(-hw, hh, 0.0, 1.0);

    [
        LineVertex {
            position: bl.into(),
            color: [1.0, 1.0, 1.0, 1.0],
        },
        LineVertex {
            position: br.into(),
            color: [1.0, 1.0, 1.0, 1.0],
        },
        LineVertex {
            position: tr.into(),
            color: [1.0, 1.0, 1.0, 1.0],
        },
        LineVertex {
            position: tl.into(),
            color: [1.0, 1.0, 1.0, 1.0],
        },
    ]
}
