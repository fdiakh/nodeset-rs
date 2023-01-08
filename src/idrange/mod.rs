mod rangelist;
mod rangetree;

pub use rangelist::IdRangeList;
pub use rangetree::IdRangeTree;

pub trait SortedIterator: Iterator {}

/// Interface for a 1-dimensional range of numerical ids
pub trait IdRange: From<Vec<u32>> + From<u32> {
    type SelfIter<'a>: Iterator<Item = &'a u32> + Clone
    where
        Self: 'a;
    type DifferenceIter<'a>: Iterator<Item = &'a u32> + SortedIterator
    where
        Self: 'a;
    type SymmetricDifferenceIter<'a>: Iterator<Item = &'a u32> + SortedIterator
    where
        Self: 'a;
    type IntersectionIter<'a>: Iterator<Item = &'a u32> + SortedIterator
    where
        Self: 'a;
    type UnionIter<'a>: Iterator<Item = &'a u32> + SortedIterator
    where
        Self: 'a;

    fn new() -> Self;
    /// Makes the range lazy, meaning that it will no longer be automatically sorted or deduplicated when adding elements.
    ///
    /// Ensemblist operations will fail while the range is in lazy mode.
    fn lazy(self) -> Self;

    /// Restores the range to a non-lazy state. Sorts and deduplicates the range.
    fn sort(&mut self);

    /// Returns an iterator over elements in the range that are not in other range
    ///
    /// Fails if the range is lazy. The returned iterator is sorted and deduplicated
    fn difference<'a>(&'a self, other: &'a Self) -> Self::DifferenceIter<'a>;

    /// Returns an iterator over elements in either range but not both
    ///
    /// Fails if the range is lazy. The returned iterator is sorted and deduplicateda
    fn symmetric_difference<'a>(&'a self, other: &'a Self) -> Self::SymmetricDifferenceIter<'a>;

    /// Returns an iterator over elements in both ranges
    ///
    /// Fails if the range is lazy. The returned iterator is sorted and deduplicated
    fn intersection<'a>(&'a self, other: &'a Self) -> Self::IntersectionIter<'a>;

    /// Returns an iterator over elements in either range
    ///
    /// Fails if the range is lazy. The returned iterator is sorted and deduplicated
    fn union<'a>(&'a self, other: &'a Self) -> Self::UnionIter<'a>;

    /// Returns whether the range contains the given id
    ///
    /// Fails if the range is lazy
    fn contains(&self, id: u32) -> bool;

    /// Returns an iterator over elements in the range
    fn iter(&self) -> Self::SelfIter<'_>;

    /// Returns whether the range is empty
    fn is_empty(&self) -> bool;

    /// Extends the range with the given range
    fn push(&mut self, other: &Self);

    /// Extends the range with the given range
    fn push_idrs(&mut self, other: &IdRangeStep);

    /// Returns the number of elements in the range
    fn len(&self) -> usize;

    /// Extends the range with elements from the given iterator
    fn from_sorted<'b>(indexes: impl IntoIterator<Item = &'b u32> + SortedIterator) -> Self;
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IdRangeStep {
    pub start: u32,
    pub end: u32,
    pub step: usize,
}
