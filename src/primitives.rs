use std::{cell::RefCell, num::NonZeroU64};

use cgmath::{vec3, vec4};
use itertools::Itertools;

use crate::{
    instance::Instance,
    render_context::{ArrowVertex, CircleVertex, Vertex},
};

#[derive(Clone, Debug)]
pub struct PrimitiveTemplate<V: Vertex + Sized + bytemuck::Pod> {
    vertices: Vec<V>,
    indicies: Vec<u16>,
}

pub struct Primitive<V, I>
where
    V: Vertex + Sized + bytemuck::Pod,
    I: Instance + Sized,
{
    template: PrimitiveTemplate<V>,
    instances: RefCell<Vec<I>>,
}

impl<V, I> Primitive<V, I>
where
    V: Vertex + Sized + bytemuck::Pod,
    I: Instance + Sized,
{
    pub fn new(template: PrimitiveTemplate<V>) -> Self {
        Self {
            template,
            instances: Default::default(),
        }
    }

    #[inline]
    pub fn instances(&self) -> &RefCell<Vec<I>> {
        &self.instances
    }
    #[inline]
    pub fn instances_raw(&self) -> Vec<I::Raw> {
        self.instances
            .borrow()
            .iter()
            .map(|i| i.to_raw())
            .collect_vec()
    }
    #[inline]
    pub fn add_instance(&self, instance: I) {
        self.instances.borrow_mut().push(instance);
    }
    #[inline]
    pub fn clear_instances(&self) {
        self.instances.borrow_mut().clear();
    }
    #[inline]
    pub fn has_instances(&self) -> bool {
        !self.instances.borrow().is_empty()
    }
    #[inline]
    pub fn num_instances(&self) -> u32 {
        self.instances.borrow().len() as u32
    }

    pub fn index_buffer_len(&self) -> NonZeroU64 {
        NonZeroU64::new((self.template.indicies.len() * std::mem::size_of::<u16>()) as u64).unwrap()
    }
    pub fn vertex_buffer_len(&self) -> NonZeroU64 {
        NonZeroU64::new((self.template.vertices.len() * std::mem::size_of::<V>()) as u64).unwrap()
    }
    pub fn instance_buffer_len(&self) -> NonZeroU64 {
        NonZeroU64::new((self.instances.borrow().len() * std::mem::size_of::<I::Raw>()) as u64)
            .unwrap()
    }

    #[inline]
    pub fn template(&self) -> &PrimitiveTemplate<V> {
        &self.template
    }
    #[inline]
    pub fn indicies(&self) -> &[u16] {
        &self.template.indicies
    }
    #[inline]
    pub fn num_indicies(&self) -> u32 {
        self.template.indicies.len() as u32
    }
    #[inline]
    pub fn vertices(&self) -> &[V] {
        &self.template.vertices
    }
}

/// creates an arrow primitive
pub fn create_arrow_template(
    length: f32,
    line_thickness: f32,
    point_length: f32,
    point_width: f32,
) -> PrimitiveTemplate<ArrowVertex> {
    let line_hw = line_thickness / 2.0;
    let line_length = length - point_length;
    let point_hw = point_width / 2.0;

    let line_bl = vec4(0.0, -line_hw, 0.0, 1.0);
    let line_br = vec4(line_length, -line_hw, 0.0, 1.0);
    let line_tr = vec4(line_length, line_hw, 0.0, 1.0);
    let line_tl = vec4(0.0, line_hw, 0.0, 1.0);

    let point_bottom = vec4(line_length, -point_hw, 0.0, 1.0);
    let point_tip = vec4(length, 0.0, 0.0, 1.0);
    let point_top = vec4(line_length, point_hw, 0.0, 1.0);

    #[rustfmt::skip]
    let vertices = [
        ArrowVertex { p: line_bl.into()     , c: [0.0, 0.0, 0.0, 0.0]},
        ArrowVertex { p: line_br.into()     , c: [0.0, 0.0, 0.0, 0.0]},
        ArrowVertex { p: line_tr.into()     , c: [0.0, 0.0, 0.0, 0.0]},
        ArrowVertex { p: line_tl.into()     , c: [0.0, 0.0, 0.0, 0.0]},
        ArrowVertex { p: point_bottom.into(), c: [0.0, 0.0, 0.0, 0.0]},
        ArrowVertex { p: point_tip.into()   , c: [0.0, 0.0, 0.0, 0.0]},
        ArrowVertex { p: point_top.into()   , c: [0.0, 0.0, 0.0, 0.0]}
    ].to_vec();

    let indicies = [0, 1, 2, 2, 3, 0, 4, 5, 6].to_vec();

    PrimitiveTemplate { vertices, indicies }
}

/// Create Circle Primitive
pub fn create_circle_primitive(radius: f32) -> PrimitiveTemplate<CircleVertex> {
    let rad = vec3(radius, -radius, 0.0);

    let bl = -rad.xxzz();
    let br = -rad.yxzz();
    let tr = rad.xxzz();
    let tl = rad.yxzz();

    let c = [0.0, 0.0, 0.0, 0.0];

    #[rustfmt::skip]
    let vertices = [
        CircleVertex { p: bl.into(), c, fc: [-1., -1.]},
        CircleVertex { p: br.into(), c, fc: [-1.,  1.]},
        CircleVertex { p: tr.into(), c, fc: [ 1.,  1.]},
        CircleVertex { p: tl.into(), c, fc: [ 1., -1.]},
    ].to_vec();
    let indicies = [0, 1, 2, 2, 3, 0].to_vec();

    PrimitiveTemplate { vertices, indicies }
}
