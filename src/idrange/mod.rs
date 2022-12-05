mod rangelist;
mod rangetree;

pub use rangelist::IdRangeList;
pub use rangetree::IdRangeTree;

pub trait IdRange {
    type SelfIter<'a>: Iterator<Item = &'a u32> + Clone
    where
        Self: 'a;
    type DifferenceIter<'a>: Iterator<Item = &'a u32>
    where
        Self: 'a;
    type SymmetricDifferenceIter<'a>: Iterator<Item = &'a u32>
    where
        Self: 'a;
    type IntersectionIter<'a>: Iterator<Item = &'a u32>
    where
        Self: 'a;
    type UnionIter<'a>: Iterator<Item = &'a u32>
    where
        Self: 'a;

    fn new(indexes: Vec<u32>) -> Self;
    fn new_empty() -> Self;
    fn difference<'a>(&'a self, other: &'a Self) -> Self::DifferenceIter<'a>;
    fn symmetric_difference<'a>(&'a self, other: &'a Self) -> Self::SymmetricDifferenceIter<'a>;
    fn intersection<'a>(&'a self, other: &'a Self) -> Self::IntersectionIter<'a>;
    fn union<'a>(&'a self, other: &'a Self) -> Self::UnionIter<'a>;
    fn iter(&self) -> Self::SelfIter<'_>;
    fn contains(&self, id: u32) -> bool;
    fn is_empty(&self) -> bool;
    fn push(&mut self, other: &Self);
    fn push_idrs(&mut self, other: &IdRangeStep);
    fn len(&self) -> usize;
    fn sort(&mut self);
    fn force_sorted(self) -> Self;
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IdRangeStep {
    pub start: u32,
    pub end: u32,
    pub step: usize,
}
