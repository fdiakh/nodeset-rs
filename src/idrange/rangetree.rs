use super::SortedIterator;
use super::{IdRange, IdRangeStep};
use itertools::Itertools;
use std::collections::btree_set;
use std::collections::BTreeSet;
use std::fmt::{self, Debug, Display};

impl From<u32> for IdRangeTree {
    fn from(index: u32) -> Self {
        let mut bt = BTreeSet::new();
        bt.insert(index);
        IdRangeTree { indexes: bt }
    }
}

impl From<Vec<u32>> for IdRangeTree {
    fn from(indexes: Vec<u32>) -> Self {
        let mut bt = BTreeSet::new();
        bt.extend(&indexes);
        IdRangeTree { indexes: bt }
    }
}

impl SortedIterator for btree_set::Union<'_, u32> {}
impl SortedIterator for btree_set::Difference<'_, u32> {}
impl SortedIterator for btree_set::Intersection<'_, u32> {}
impl SortedIterator for btree_set::SymmetricDifference<'_, u32> {}


impl IdRange for IdRangeTree {
    type SelfIter<'a> = btree_set::Iter<'a, u32>;
    type DifferenceIter<'a> = btree_set::Difference<'a, u32>;
    type SymmetricDifferenceIter<'a> = btree_set::SymmetricDifference<'a, u32>;
    type IntersectionIter<'a> = btree_set::Intersection<'a, u32>;
    type UnionIter<'a> = btree_set::Union<'a, u32>;

    fn from_sorted<'b>(indexes: impl IntoIterator<Item = &'b u32>) -> IdRangeTree {
        let mut bt = BTreeSet::new();
        bt.extend(indexes);
        IdRangeTree { indexes: bt }
    }

    fn new() -> Self {
        IdRangeTree {
            indexes: BTreeSet::new(),
        }
    }

    fn difference<'a>(&'a self, other: &'a Self) -> Self::DifferenceIter<'a> {
        self.indexes.difference(&other.indexes)
    }

    fn symmetric_difference<'a>(&'a self, other: &'a Self) -> Self::SymmetricDifferenceIter<'a> {
        self.indexes.symmetric_difference(&other.indexes)
    }

    fn intersection<'a>(&'a self, other: &'a Self) -> Self::IntersectionIter<'a> {
        self.indexes.intersection(&other.indexes)
    }
    fn union<'a>(&'a self, other: &'a Self) -> Self::UnionIter<'a> {
        self.indexes.union(&other.indexes)
    }
    fn iter(&self) -> Self::SelfIter<'_> {
        self.indexes.iter()
    }
    fn contains(&self, id: u32) -> bool {
        self.indexes.contains(&id)
    }
    fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }
    fn push(&mut self, other: &Self) {
        self.indexes.extend(&other.indexes);
    }
    fn push_idrs(&mut self, idrs: &IdRangeStep) {
        self.indexes
            .extend((idrs.start..idrs.end + 1).step_by(idrs.step))
    }
    fn len(&self) -> usize {
        self.indexes.len()
    }
    fn sort(&mut self) {}
    fn lazy(self) -> Self {
        self
    }
}

impl Display for IdRangeTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut rngs = self
            .indexes
            .iter()
            .zip(
                self.indexes
                    .iter()
                    .chain(self.indexes.iter().last())
                    .skip(1),
            )
            .batching(|it| {
                if let Some((&first, &next)) = it.next() {
                    if next != first + 1 {
                        return Some(first.to_string());
                    }
                    for (&cur, &next) in it {
                        if next != cur + 1 {
                            return Some(format!("{}-{}", first, cur));
                        }
                    }
                    // Should never be reached
                    None
                } else {
                    None
                }
            });
        write!(f, "[{}]", rngs.join(","))
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IdRangeTree {
    indexes: BTreeSet<u32>,
}
