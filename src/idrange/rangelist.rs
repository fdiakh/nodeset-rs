use super::{CachedTranslation, IdRange, IdRangeStep, RankRanges, SortedIterator};

use std::fmt::{self, Debug, Display};

/// A 1D set of indexes stored in a Vec
#[derive(Debug, Clone)]
pub struct IdRangeList {
    indexes: Vec<u32>,
    sorted: bool,
}

impl PartialEq for IdRangeList {
    fn eq(&self, other: &Self) -> bool {
        self.indexes == other.indexes
    }
}

pub struct VecDifference<'a, T> {
    a: std::slice::Iter<'a, T>,
    b: std::iter::Peekable<std::slice::Iter<'a, T>>,
}

pub struct VecIntersection<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

pub struct VecUnion<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

pub struct VecSymDifference<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

impl IdRangeList {}

impl SortedIterator for VecUnion<'_, u32> {}
impl SortedIterator for VecIntersection<'_, u32> {}
impl SortedIterator for VecDifference<'_, u32> {}
impl SortedIterator for VecSymDifference<'_, u32> {}

impl<'a, T> Iterator for VecDifference<'a, T>
where
    T: Ord + Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next = match self.a.next() {
            Some(v) => v,
            None => return None,
        };

        let mut min: &T;
        loop {
            min = match self.b.peek() {
                None => return Some(next),
                Some(v) if *v == next => v,
                Some(v) if *v > next => return Some(next),
                _ => {
                    self.b.next();
                    continue;
                }
            };

            while next == min {
                next = match self.a.next() {
                    Some(v) => v,
                    None => return None,
                };
            }
        }
    }
}

impl<'a, T> Iterator for VecIntersection<'a, T>
where
    T: Ord + Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.a.is_empty() && !self.b.is_empty() {
            let first_a = self.a.first();
            let first_b = self.b.first();

            match first_a.cmp(&first_b) {
                std::cmp::Ordering::Less => {
                    self.a = &self.a[exponential_search_idx(self.a, first_b.unwrap())..];
                }

                std::cmp::Ordering::Equal => {
                    self.a = &self.a[1..];
                    self.b = &self.b[1..];
                    return first_a;
                }
                std::cmp::Ordering::Greater => {
                    self.b = &self.b[exponential_search_idx(self.b, first_a.unwrap())..];
                }
            }
        }

        None
    }
}

impl<'a, T> Iterator for VecUnion<'a, T>
where
    T: Ord + Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let first_a = self.a.first();
        let first_b = self.b.first();

        if first_a.is_some() && (first_a <= first_b || first_b.is_none()) {
            self.a = &self.a[1..];
            return first_a;
        }

        if first_b.is_some() {
            self.b = &self.b[1..];
            return first_b;
        }

        None
    }
}

impl<'a, T> Iterator for VecSymDifference<'a, T>
where
    T: Ord + Debug,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut first_a = self.a.first();
        let mut first_b = self.b.first();

        while first_a == first_b {
            first_a?;
            self.a = &self.a[1..];
            self.b = &self.b[1..];
            first_a = self.a.first();
            first_b = self.b.first();
        }

        if first_a.is_some() && (first_a < first_b || first_b.is_none()) {
            self.a = &self.a[1..];
            return first_a;
        }
        self.b = &self.b[1..];
        first_b
    }
}

fn exponential_search<T>(v: &[T], x: &T) -> Result<usize, usize>
where
    T: Ord,
{
    let mut i: usize = 1;
    while i <= v.len() {
        if v[i - 1] == *x {
            return Ok(i - 1);
        }
        if v[i - 1] > *x {
            break;
        }
        i *= 2;
    }

    match v[i / 2..std::cmp::min(i - 1, v.len())].binary_search(x) {
        Ok(result) => Ok(result + i / 2),
        Err(result) => Err(result + i / 2),
    }
}

fn exponential_search_idx<T>(v: &[T], x: &T) -> usize
where
    T: Ord,
{
    match exponential_search(v, x) {
        Ok(r) => r,
        Err(r) => r,
    }
}

impl From<u32> for IdRangeList {
    fn from(index: u32) -> IdRangeList {
        IdRangeList {
            indexes: vec![index],
            sorted: true,
        }
    }
}

impl From<Vec<u32>> for IdRangeList {
    fn from(indexes: Vec<u32>) -> IdRangeList {
        let mut r = IdRangeList {
            indexes,
            sorted: false,
        };
        r.sort();
        r
    }
}

impl From<IdRangeStep> for IdRangeList {
    fn from(idrs: IdRangeStep) -> IdRangeList {
        let mut r = IdRangeList {
            indexes: vec![],
            sorted: true,
        };
        r.push_idrs(idrs);
        r
    }
}

impl IdRange for IdRangeList {
    type DifferenceIter<'a> = VecDifference<'a, u32>;
    type IntersectionIter<'a> = VecIntersection<'a, u32>;
    type UnionIter<'a> = VecUnion<'a, u32>;
    type SymmetricDifferenceIter<'a> = VecSymDifference<'a, u32>;
    type SelfIter<'a> = std::slice::Iter<'a, u32>;
    fn from_sorted<'b>(indexes: impl IntoIterator<Item = &'b u32>) -> IdRangeList {
        IdRangeList {
            indexes: indexes.into_iter().cloned().collect(),
            sorted: true,
        }
    }

    fn new() -> Self {
        IdRangeList {
            indexes: vec![],
            sorted: true,
        }
    }
    fn push_idrs(&mut self, ranges: impl RankRanges) {
        let sorted_after = self.sorted && ranges.start_rank() > *self.indexes.last().unwrap_or(&0);

        for (start, end, step) in ranges.rank_ranges() {
            if end != u32::MAX {
                self.indexes.extend((start..end + 1).step_by(step as usize));
            } else {
                // For some reason extending using an iterator built with ..= is
                // much slower so we only use it for u32::MAX
                self.indexes
                    .extend((start..=u32::MAX).step_by(step as usize));
            }
        }

        if self.sorted && !sorted_after {
            self.sorted = false;
            self.sort();
        }
    }

    fn len(&self) -> usize {
        self.indexes.len()
    }

    fn sort(&mut self) {
        if !self.sorted {
            self.indexes.sort_unstable();
            self.indexes.dedup();
            self.sorted = true;
        }
    }

    fn push(&mut self, other: &Self) {
        let sorted =
            self.sorted && other.sorted && other.indexes[0] > *self.indexes.last().unwrap_or(&0);
        self.indexes.extend(other.iter());
        if self.sorted && !sorted {
            self.sorted = false;
            self.sort();
        }
    }

    fn contains(&self, id: u32) -> bool {
        assert!(self.sorted);

        exponential_search(&self.indexes, &id).is_ok()
    }

    fn iter(&self) -> Self::SelfIter<'_> {
        self.indexes.iter()
    }

    fn intersection<'a>(&'a self, other: &'a Self) -> Self::IntersectionIter<'a> {
        assert!(self.sorted);
        assert!(other.sorted);

        Self::IntersectionIter {
            a: &self.indexes,
            b: &other.indexes,
        }
    }

    fn union<'a>(&'a self, other: &'a Self) -> Self::UnionIter<'a> {
        assert!(self.sorted);
        assert!(other.sorted);

        Self::UnionIter {
            a: &self.indexes,
            b: &other.indexes,
        }
    }

    fn symmetric_difference<'a>(&'a self, other: &'a Self) -> Self::SymmetricDifferenceIter<'a> {
        assert!(self.sorted);
        assert!(other.sorted);

        Self::SymmetricDifferenceIter {
            a: &self.indexes,
            b: &other.indexes,
        }
    }

    fn difference<'a>(&'a self, other: &'a Self) -> Self::DifferenceIter<'a> {
        assert!(self.sorted);
        assert!(other.sorted);

        Self::DifferenceIter {
            a: self.indexes.iter(),
            b: other.indexes.iter().peekable(),
        }
    }

    fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    fn lazy(mut self) -> Self {
        self.sorted = false;
        self
    }

    fn set_lazy(&mut self) {
        self.sorted = false;
    }
}
use std::iter;

impl Display for IdRangeList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            return fmt::Result::Ok(());
        }

        if self.len() == 1 {
            return f.write_str(&CachedTranslation::new(self.indexes[0]).to_string());
        }

        let ranges = super::fold_into_ranges(
            self.indexes
                .iter()
                .chain(iter::once(&self.indexes[self.indexes.len() - 1])),
            self.indexes[0],
        );

        if f.alternate() {
            write!(f, "{}", ranges)
        } else {
            write!(f, "[{}]", ranges)
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_exponential_search() {
        assert_eq!(exponential_search(&[], &0), Err(0));
        assert_eq!(exponential_search(&[1, 2, 4, 7], &4), Ok(2));
        assert_eq!(exponential_search(&[1, 2, 4, 7], &0), Err(0));
        assert_eq!(exponential_search(&[1, 2, 4, 7], &8), Err(4));
    }

    fn validate_rangelist_union_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>) {
        let rl1 = IdRangeList {
            indexes: a,
            sorted: true,
        };

        let rl2 = IdRangeList {
            indexes: b,
            sorted: true,
        };
        assert_eq!(rl1.union(&rl2).copied().collect::<Vec<u32>>(), c);
    }

    fn validate_rangelist_symdiff_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>) {
        let rl1 = IdRangeList {
            indexes: a,
            sorted: true,
        };

        let rl2 = IdRangeList {
            indexes: b,
            sorted: true,
        };
        assert_eq!(
            rl1.symmetric_difference(&rl2)
                .cloned()
                .collect::<Vec<u32>>(),
            c
        );
    }

    fn validate_rangelist_intersection_result(a: Vec<u32>, b: Vec<u32>, c: Vec<u32>) {
        let rl1 = IdRangeList {
            indexes: a,
            sorted: true,
        };

        let rl2 = IdRangeList {
            indexes: b,
            sorted: true,
        };
        assert_eq!(rl1.intersection(&rl2).cloned().collect::<Vec<u32>>(), c);
    }
    #[test]
    fn rangelist_union() {
        validate_rangelist_union_result(vec![0, 4, 9], vec![1, 2, 5, 7], vec![0, 1, 2, 4, 5, 7, 9]);
        validate_rangelist_union_result(vec![], vec![1, 2, 5, 7], vec![1, 2, 5, 7]);
        validate_rangelist_union_result(vec![0, 4, 9], vec![], vec![0, 4, 9]);
        validate_rangelist_union_result(vec![0, 4, 9], vec![10, 11, 12], vec![0, 4, 9, 10, 11, 12]);
    }

    #[test]
    fn rangelist_symdiff() {
        validate_rangelist_symdiff_result(
            vec![0, 2, 4, 7, 9],
            vec![1, 2, 5, 7],
            vec![0, 1, 4, 5, 9],
        );
        validate_rangelist_symdiff_result(vec![], vec![1, 2, 5, 7], vec![1, 2, 5, 7]);
        validate_rangelist_symdiff_result(vec![0, 4, 9], vec![], vec![0, 4, 9]);
        validate_rangelist_symdiff_result(
            vec![0, 4, 9],
            vec![10, 11, 12],
            vec![0, 4, 9, 10, 11, 12],
        );
    }

    #[test]
    fn rangelist_intersection() {
        validate_rangelist_intersection_result(vec![0, 4, 9], vec![1, 2, 5, 7], vec![]);
        validate_rangelist_intersection_result(vec![], vec![1, 2, 5, 7], vec![]);
        validate_rangelist_intersection_result(vec![0, 4, 9], vec![], vec![]);
        validate_rangelist_intersection_result(
            vec![0, 4, 9, 7, 12, 34, 35],
            vec![4, 11, 12, 37],
            vec![4, 12],
        );
        validate_rangelist_intersection_result(
            vec![4, 11, 12, 37],
            vec![0, 4, 9, 7, 12, 34, 35],
            vec![4, 12],
        );
    }

    #[test]
    fn rangelist_difference() {
        let rl1 = IdRangeList {
            indexes: vec![1, 2, 3],
            sorted: true,
        };

        let mut rl2 = IdRangeList {
            indexes: vec![1, 3],
            sorted: true,
        };
        assert_eq!(rl1.difference(&rl2).cloned().collect::<Vec<u32>>(), vec![2]);

        rl2 = IdRangeList {
            indexes: vec![],
            sorted: true,
        };
        assert_eq!(
            rl1.difference(&rl2).cloned().collect::<Vec<u32>>(),
            vec![1, 2, 3]
        );
        assert_eq!(rl2.difference(&rl1).cloned().collect::<Vec<u32>>(), vec![]);
        rl2 = IdRangeList {
            indexes: vec![4, 5, 6],
            sorted: true,
        };
        assert_eq!(
            rl1.difference(&rl2).cloned().collect::<Vec<u32>>(),
            vec![1, 2, 3]
        );
        assert_eq!(rl1.difference(&rl1).cloned().collect::<Vec<u32>>(), vec![]);
    }

    #[test]
    #[should_panic]
    fn rangelist_difference_bad1() {
        let rl1 = IdRangeList {
            indexes: vec![1, 2, 3],
            sorted: true,
        };

        let rl2 = IdRangeList {
            indexes: vec![3, 1],
            sorted: false,
        };

        rl1.difference(&rl2);
    }
}
