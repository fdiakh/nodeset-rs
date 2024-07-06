mod rangelist;
mod rangetree;

pub use rangelist::IdRangeList;
pub use rangetree::IdRangeTree;
use std::fmt;

/// Lookup table for powers of 10 that can be represented as u32
const POW10_LOOKUP: [u32; 10] = [
    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000,
];

/// Iterators implementing this trait guarantee that their elements are sorted are deduplicated
pub trait SortedIterator: Iterator {}

/// Interface for a 1-dimensional range of integers
pub trait IdRange: From<Vec<u32>> + From<u32> {
    type SelfIter<'a>: Iterator<Item = &'a u32> + Clone + fmt::Debug
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

    fn set_lazy(&mut self);

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

    /// Extends the range with the given contiguous range of zero-padded ids
    fn push_idrs(&mut self, other: &IdRangeStep);

    /// Returns the number of elements in the range
    fn len(&self) -> usize;

    /// Extends the range with elements from the given iterator
    fn from_sorted<'b>(indexes: impl IntoIterator<Item = &'b u32> + SortedIterator) -> Self;
}

#[derive(Debug, PartialEq, Eq, Clone)]
/// A contiguous range of zero-padded ids
///
/// A contiguous range of ids does not necessarily translate to a contiguous
/// range of ranks
pub struct IdRangeStep {
    pub start: u32,
    pub end: u32,
    pub step: usize,
    /// The minimum number of digits in the zero-padded ids: a start of 2 and a
    /// pad of 3 means that the range starts at 002
    pub pad: u32,
}

impl IdRangeStep {
    /// Returns the rank of the first id in the range
    /// Zero-padded ids are sorted such as 1 < 9 < 00 < 09 < 10 < 99 < 000 ...
    fn start_rank(&self) -> u32 {
        padded_id_to_rank(self.start, self.pad)
    }

    /// Convert a zero-padded id range into a list of contiguous rank ranges.
    /// While it would be more elegant to return an iterator here but it would
    /// usually only yield very few elements
    fn rank_ranges(&self) -> Vec<(u32, u32, usize)> {
        // FIXME: overflow handling is not right if we hit the saturating point.
        // We should prevent creation of idranges that are too large
        let mut res = Vec::new();
        let mut bound = POW10_LOOKUP[u32::min(9, u32::max(self.pad, 1)) as usize];

        let mut start = if self.start < bound {
            let remainder = (bound - 1 - self.start) % self.step as u32;

            res.push((
                padded_id_to_rank(self.start, self.pad),
                padded_id_to_rank(u32::min(self.end, bound - 1), self.pad),
                self.step,
            ));
            let prev_bound = bound;
            bound = bound.saturating_mul(10);
            prev_bound + (self.step as u32) - remainder - 1
        } else {
            bound = lower_pow10_bound(self.start).0.saturating_mul(10);
            self.start
        };

        while start <= self.end {
            let remainder = (bound - 1 - self.start) % self.step as u32;

            let end = u32::min(self.end, bound - 1);
            res.push((
                padded_id_to_rank(start, self.pad),
                padded_id_to_rank(end, self.pad),
                self.step,
            ));
            start = bound + self.step as u32 - remainder - 1;
            bound = bound.saturating_mul(10);
        }

        res
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
/// Optimizes the translation of a rank to a zero-padded id by caching
/// costly intermediate computations
pub(crate) struct CachedTranslation {
    rank: u32,
    id: u32,
    pad: u32,
    jump_pad: u32,
}

impl CachedTranslation {
    /// Returns the maximum padded length of ids that can be merged with this one
    /// in a contiguous range
    fn max_pad(&self) -> u32 {
        if self.rank != 0 && self.id < self.jump_pad / 10 {
            self.pad
        } else {
            u32::MAX
        }
    }

    pub fn padding(&self) -> u32 {
        self.pad
    }

    /// Maps a rank to a zero-padded id and returns it along with cached
    /// values
    pub(crate) fn new(rank: u32) -> Self {
        let mut id = rank;
        let mut pad = 1;
        let mut jump_pad = 10;

        while id >= jump_pad {
            id -= jump_pad;
            jump_pad *= 10;
            pad += 1;
        }

        CachedTranslation {
            rank,
            id,
            pad,
            jump_pad,
        }
    }

    /// Maps a rank to a zero-padded id using cached values if possible
    /// Results are only valid for ranks greater than the rank used to create the cache
    pub(crate) fn interpolate(&self, new_rank: u32) -> Self {
        let jump_rank = self.rank + self.jump_pad - self.id;
        if new_rank < jump_rank {
            return Self {
                rank: new_rank,
                id: self.id + (new_rank - self.rank),
                pad: self.pad,
                jump_pad: self.jump_pad,
            };
        }
        CachedTranslation::new(new_rank)
    }

    /// Returns whether the given rank can be merged with this one while meeting
    /// the max_pad constraint
    fn is_mergeable(&self, other: &Self, max_pad: u32) -> bool {
        other.id == self.id + 1 && other.pad <= max_pad
    }
}

/// Display the cached rank as a zero-padded string
impl fmt::Display for CachedTranslation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        let pad = self.pad as usize;
        let id = self.id;

        if id >= self.jump_pad / 10 {
            write!(f, "{id}")
        } else {
            write!(f, "{id:0>pad$}")
        }
    }
}

/// Converts a padded id to a rank
///
/// # Examples
///
/// assert_eq!(padded_id_to_rank(423, 4), 1533);
fn padded_id_to_rank(id: u32, pad: u32) -> u32 {
    let mut rank: u32 = 0;

    for _ in 1..u32::max(lower_pow10_bound(id).1 + 1, pad) {
        rank = rank * 10 + 10
    }

    id + rank
}

/// Returns the closest power of 10 that is less than or equal to n and the corresponding exponent
///
/// # Examples
///
/// assert_eq!(lower_pow10_bound(423), (100, 2));
fn lower_pow10_bound(n: u32) -> (u32, u32) {
    if n == 0 {
        return (1, 0);
    }

    let exp = u32::ilog10(n);
    let power = POW10_LOOKUP[exp as usize];

    (power, exp)
}

/// Converts a list of ranks into a comma-separated string of contiguous ranges
//  FIXME: this should take a sorted iterator for safety
fn fold_into_ranges<'a>(iter: impl Iterator<Item = &'a u32>, first_rank: u32) -> String {
    use itertools::Itertools;

    let mut cache = CachedTranslation::new(first_rank);

    let mut rngs = iter.skip(1).batching(|it| {
        if let Some(&next) = it.next() {
            let max_pad = cache.max_pad();
            let mut new_cache = cache.interpolate(next);
            let mergeable = cache.is_mergeable(&new_cache, max_pad);

            if !mergeable {
                let res = cache.to_string();
                cache = new_cache;
                return Some(res);
            }

            let mut cur_cache = new_cache;
            for &next in it {
                new_cache = cur_cache.interpolate(next);
                let mergeable = cur_cache.is_mergeable(&new_cache, max_pad);

                if !mergeable {
                    let res = format!("{cache}-{cur_cache}");
                    cache = new_cache;
                    return Some(res);
                } else {
                    std::mem::swap(&mut cur_cache, &mut new_cache);
                }
            }
            // Should never be reached
            None
        } else {
            None
        }
    });

    rngs.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Displays a rank as a zero-padded string
    fn rank_to_string(rank: u32) -> String {
        CachedTranslation::new(rank).to_string()
    }

    #[test]
    fn test_rank_ranges() {
        assert_eq!(
            IdRangeStep {
                start: 0,
                end: 9,
                step: 1,
                pad: 0
            }
            .rank_ranges()
            .iter()
            .map(|(start, end, step)| (rank_to_string(*start), rank_to_string(*end), *step))
            .collect::<Vec<_>>(),
            vec![("0".to_string(), "9".to_string(), 1)]
        );

        assert_eq!(
            IdRangeStep {
                start: 100,
                end: 110,
                step: 1,
                pad: 0
            }
            .rank_ranges()
            .iter()
            .map(|(start, end, step)| (rank_to_string(*start), rank_to_string(*end), *step))
            .collect::<Vec<_>>(),
            vec![("100".to_string(), "110".to_string(), 1)]
        );

        assert_eq!(
            IdRangeStep {
                start: 0,
                end: 20,
                step: 1,
                pad: 1
            }
            .rank_ranges()
            .iter()
            .map(|(start, end, step)| (rank_to_string(*start), rank_to_string(*end), *step))
            .collect::<Vec<_>>(),
            vec![
                ("0".to_string(), "9".to_string(), 1),
                ("10".to_string(), "20".to_string(), 1)
            ]
        );

        assert_eq!(
            IdRangeStep {
                start: 0,
                end: 20,
                step: 1,
                pad: 2
            }
            .rank_ranges()
            .iter()
            .map(|(start, end, step)| (rank_to_string(*start), rank_to_string(*end), *step))
            .collect::<Vec<_>>(),
            vec![("00".to_string(), "20".to_string(), 1)]
        );

        assert_eq!(
            IdRangeStep {
                start: 0,
                end: 15,
                step: 7,
                pad: 0
            }
            .rank_ranges()
            .iter()
            .map(|(start, end, step)| (rank_to_string(*start), rank_to_string(*end), *step))
            .collect::<Vec<_>>(),
            vec![
                ("0".to_string(), "9".to_string(), 7),
                ("14".to_string(), "15".to_string(), 7)
            ]
        );

        assert_eq!(
            IdRangeStep {
                start: 0,
                end: 1500,
                step: 7,
                pad: 2
            }
            .rank_ranges()
            .iter()
            .map(|(start, end, step)| (rank_to_string(*start), rank_to_string(*end), *step))
            .collect::<Vec<_>>(),
            vec![
                ("00".to_string(), "99".to_string(), 7),
                ("105".to_string(), "999".to_string(), 7),
                ("1001".to_string(), "1500".to_string(), 7)
            ]
        );
    }

    use std::num::ParseIntError;
    fn rank_of_string(s: &str) -> Result<u32, ParseIntError> {
        Ok(padded_id_to_rank(s.parse::<u32>()?, s.len() as u32))
    }

    #[test]
    fn test_rank_of_string() {
        assert_eq!(rank_of_string("0").unwrap(), 0);
        assert_eq!(rank_of_string("9").unwrap(), 9);
        assert_eq!(rank_of_string("01").unwrap(), 11);
        assert_eq!(rank_of_string("09").unwrap(), 19);
        assert_eq!(rank_of_string("10").unwrap(), 20);
        assert_eq!(rank_of_string("19").unwrap(), 29);
        assert_eq!(rank_of_string("99").unwrap(), 109);
        assert_eq!(rank_of_string("000").unwrap(), 110);
        assert_eq!(rank_of_string("001").unwrap(), 111);
        assert_eq!(rank_of_string("010").unwrap(), 120);
        assert_eq!(rank_of_string("999").unwrap(), 1109);
    }

    #[test]
    fn test_string_of_rank() {
        assert_eq!("0", rank_to_string(0));
        assert_eq!("9", rank_to_string(9));
        assert_eq!("01", rank_to_string(11));
        assert_eq!("09", rank_to_string(19));
        assert_eq!("10", rank_to_string(20));
        assert_eq!("19", rank_to_string(29));
        assert_eq!("99", rank_to_string(109));
        assert_eq!("000", rank_to_string(110));
        assert_eq!("001", rank_to_string(111));
        assert_eq!("010", rank_to_string(120));
        assert_eq!("999", rank_to_string(1109));
    }
}
