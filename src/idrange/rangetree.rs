use super::SortedIterator;
use super::{CachedTranslation, IdRange, IdRangeStep};
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
        for (start, end, step) in idrs.rank_ranges() {
            self.indexes.extend((start..end + 1).step_by(step));
        }
    }
    fn len(&self) -> usize {
        self.indexes.len()
    }
    fn sort(&mut self) {}
    fn lazy(self) -> Self {
        self
    }

    fn set_lazy(&mut self) {}
}

impl Display for IdRangeTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            return fmt::Result::Ok(());
        }

        if self.len() == 1 {
            return f
                .write_str(&CachedTranslation::new(*self.indexes.first().unwrap()).to_string());
        }

        let ranges = super::fold_into_ranges(
            self.indexes.iter().chain(self.indexes.last()),
            *self.indexes.first().unwrap(),
        );

        if f.alternate() {
            write!(f, "{}", ranges)
        } else {
            write!(f, "[{}]", ranges)
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IdRangeTree {
    indexes: BTreeSet<u32>,
}

#[cfg(test)]
pub mod tests {
    use super::*;

    //create a btreeset from a vec<u32>
    fn bt_from_vec(v: Vec<u32>) -> BTreeSet<u32> {
        let mut bt = BTreeSet::new();
        bt.extend(v);
        bt
    }

    fn validate_rangetree_union_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>) {
        let rl1 = IdRangeTree {
            indexes: bt_from_vec(a),
        };

        let rl2 = IdRangeTree {
            indexes: bt_from_vec(b),
        };
        assert_eq!(rl1.union(&rl2).copied().collect::<Vec<u32>>(), c);
    }

    fn validate_rangetree_symdiff_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>) {
        let rl1 = IdRangeTree {
            indexes: bt_from_vec(a),
        };

        let rl2 = IdRangeTree {
            indexes: bt_from_vec(b),
        };
        assert_eq!(
            rl1.symmetric_difference(&rl2)
                .cloned()
                .collect::<Vec<u32>>(),
            c
        );
    }

    fn validate_rangetree_intersection_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>) {
        let rl1 = IdRangeTree {
            indexes: bt_from_vec(a),
        };

        let rl2 = IdRangeTree {
            indexes: bt_from_vec(b),
        };
        assert_eq!(rl1.intersection(&rl2).cloned().collect::<Vec<u32>>(), c);
    }
    #[test]
    fn rangetree_union() {
        validate_rangetree_union_result(vec![0, 4, 9], vec![1, 2, 5, 7], vec![0, 1, 2, 4, 5, 7, 9]);
        validate_rangetree_union_result(vec![], vec![1, 2, 5, 7], vec![1, 2, 5, 7]);
        validate_rangetree_union_result(vec![0, 4, 9], vec![], vec![0, 4, 9]);
        validate_rangetree_union_result(vec![0, 4, 9], vec![10, 11, 12], vec![0, 4, 9, 10, 11, 12]);
    }

    #[test]
    fn rangetree_symdiff() {
        validate_rangetree_symdiff_result(
            vec![0, 2, 4, 7, 9],
            vec![1, 2, 5, 7],
            vec![0, 1, 4, 5, 9],
        );
        validate_rangetree_symdiff_result(vec![], vec![1, 2, 5, 7], vec![1, 2, 5, 7]);
        validate_rangetree_symdiff_result(vec![0, 4, 9], vec![], vec![0, 4, 9]);
        validate_rangetree_symdiff_result(
            vec![0, 4, 9],
            vec![10, 11, 12],
            vec![0, 4, 9, 10, 11, 12],
        );
    }

    #[test]
    fn rangetree_intersection() {
        validate_rangetree_intersection_result(vec![0, 4, 9], vec![1, 2, 5, 7], vec![]);
        validate_rangetree_intersection_result(vec![], vec![1, 2, 5, 7], vec![]);
        validate_rangetree_intersection_result(vec![0, 4, 9], vec![], vec![]);
        validate_rangetree_intersection_result(
            vec![0, 4, 9, 7, 12, 34, 35],
            vec![4, 11, 12, 37],
            vec![4, 12],
        );
        validate_rangetree_intersection_result(
            vec![4, 11, 12, 37],
            vec![0, 4, 9, 7, 12, 34, 35],
            vec![4, 12],
        );
    }

    #[test]
    fn rangetree_difference() {
        let rl1 = IdRangeTree {
            indexes: bt_from_vec(vec![1, 2, 3]),
        };

        let mut rl2 = IdRangeTree {
            indexes: bt_from_vec(vec![1, 3]),
        };
        assert_eq!(rl1.difference(&rl2).cloned().collect::<Vec<u32>>(), vec![2]);

        rl2 = IdRangeTree {
            indexes: bt_from_vec(vec![]),
        };
        assert_eq!(
            rl1.difference(&rl2).cloned().collect::<Vec<u32>>(),
            vec![1, 2, 3]
        );
        assert_eq!(rl2.difference(&rl1).cloned().collect::<Vec<u32>>(), vec![]);
        rl2 = IdRangeTree {
            indexes: bt_from_vec(vec![4, 5, 6]),
        };
        assert_eq!(
            rl1.difference(&rl2).cloned().collect::<Vec<u32>>(),
            vec![1, 2, 3]
        );
        assert_eq!(rl1.difference(&rl1).cloned().collect::<Vec<u32>>(), vec![]);
    }
}
