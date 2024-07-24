mod rangelist;
mod rangetree;

pub use rangelist::IdRangeList;
pub use rangetree::IdRangeTree;
use std::{error::Error, fmt};

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

    /// Extends the range with another IdRange
    fn push(&mut self, other: &Self);

    /// Extends the range with intervals of ranks
    fn push_idrs(&mut self, ranges: impl RankRanges);

    /// Returns the number of elements in the range
    fn len(&self) -> usize;

    /// Extends the range with elements from the given iterator
    fn from_sorted<'b>(indexes: impl IntoIterator<Item = &'b u32> + SortedIterator) -> Self;
}

#[derive(Debug, PartialEq, Eq, Clone)]
/// A contiguous range of zero-padded ids with optional step
///
/// A contiguous range of ids does not necessarily translate to a contiguous
/// range of ranks
pub(crate) struct IdRangeStep {
    pub start: u32,
    pub end: u32,
    pub step: u32,
    /// The minimum number of digits in the zero-padded ids: a start of 2 and a
    /// pad of 3 means that the range starts at 002
    pub pad: u32,
}

#[derive(Debug, PartialEq, Clone)]
/// Errors that may happen when defining a range
pub enum RangeStepError {
    /// A range is inverted (ie `[9-2]`)
    Reverse,
    /// An id has a rank that is larger than what can be represented by a u32
    Overflow,
}

impl Error for RangeStepError {}

impl fmt::Display for RangeStepError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RangeStepError::Reverse => write!(f, "start id is greater than end id"),
            RangeStepError::Overflow => write!(f, "id is too large"),
        }
    }
}

/// A trait for ranges of ids that can be converted to a list of rank ranges
pub trait RankRanges {
    /// Returns a list of rank ranges (start, end, step)
    fn rank_ranges(&self) -> Vec<(u32, u32, u32)>;
    /// Returns the rank of the first id in the range
    fn start_rank(&self) -> u32;
}

impl RankRanges for IdRangeStep {
    /// Returns the rank of the first id in the range
    /// Zero-padded ids are sorted such as 1 < 9 < 00 < 09 < 10 < 99 < 000 ...
    fn start_rank(&self) -> u32 {
        padded_id_to_rank_exact(self.start, self.pad)
    }

    /// Convert a zero-padded id range into a list of contiguous rank ranges.
    /// While it would be more elegant to return an iterator here but it would
    /// usually only yield very few elements
    fn rank_ranges(&self) -> Vec<(u32, u32, u32)> {
        let mut res = Vec::new();
        let mut start = self.start;
        let mut bound = pow10(self.pad);
        let mut start_pad = self.pad;

        while start <= self.end {
            if start >= bound {
                bound = bound.saturating_mul(10);
                start_pad += 1;

                continue;
            }

            let remainder = (bound - 1 - self.start) % self.step;

            let end = u32::min(self.end, bound - 1);
            res.push((
                padded_id_to_rank_exact(start, start_pad),
                padded_id_to_rank_exact(end, start_pad),
                self.step,
            ));

            start = match bound.checked_add(self.step - remainder - 1) {
                Some(s) => s,
                None => break,
            };
        }

        res
    }
}

impl IdRangeStep {
    /// Create a new IdRangeStep
    ///
    /// The pad must be at least equal to the number of digits in the start id
    pub(crate) fn new(start: u32, end: u32, step: u32, pad: u32) -> Result<Self, RangeStepError> {
        if start > end {
            return Err(RangeStepError::Reverse);
        }
        if pad > MAX_U32_PAD || start > MAX_U32_ID || end > MAX_U32_ID {
            return Err(RangeStepError::Overflow);
        }
        Ok(Self {
            start,
            end,
            step,
            pad,
        })
    }
}

/// A single zero-padded id
#[derive(Debug, PartialEq, Clone)]
pub struct SingleId {
    pub rank: u32,
}

impl SingleId {
    /// Create a new SingleId
    ///
    /// The pad must be at least equal to the number of digits in the id
    pub fn new(id: u32, pad: u32) -> Result<Self, RangeStepError> {
        if pad > MAX_U32_PAD || id > MAX_U32_ID {
            return Err(RangeStepError::Overflow);
        }

        Ok(Self {
            rank: padded_id_to_rank_exact(id, pad),
        })
    }
}

impl RankRanges for SingleId {
    fn rank_ranges(&self) -> Vec<(u32, u32, u32)> {
        vec![(self.rank, self.rank, 1)]
    }
    fn start_rank(&self) -> u32 {
        self.rank
    }
}

/// A contiguous range of zero-padded ids with optional step, prefix and suffix
#[derive(Debug, Clone, PartialEq)]
pub struct AffixIdRangeStep {
    /// The range of ids including the suffix (low order digits)
    idrs: IdRangeStep,
    /// The high order digits to add to the range
    high: IdRangeOffset,
}

impl AffixIdRangeStep {
    /// Create a new AffixIdRangeStep
    ///
    /// # Arguments
    ///
    /// * `idrs` - The range of ids (middle digits)
    /// * `low` - The low order digits to add to the range
    /// * `high` - The high order digits to add to the range
    pub fn new(
        mut idrs: IdRangeStep,
        low: Option<IdRangeOffset>,
        high: Option<IdRangeOffset>,
    ) -> Result<Self, RangeStepError> {
        if let Some(low) = low {
            idrs.pad = idrs.pad.saturating_add(low.pad);
            if idrs.pad > MAX_U32_PAD {
                return Err(RangeStepError::Overflow);
            }
            let low_bound = pow10(low.pad);
            idrs.start = idrs.start * low_bound + low.value;
            idrs.end = idrs.end.saturating_mul(low_bound).saturating_add(low.value);
            idrs.step = idrs.step.saturating_mul(low_bound);
        };

        let Some(high) = high else {
            return Ok(Self {
                idrs,
                high: IdRangeOffset::new(0, 0).unwrap(),
            });
        };

        let end_pad = u32::max(idrs.end.checked_ilog10().unwrap_or(0) + 1, idrs.pad);

        if end_pad + high.pad > MAX_U32_PAD {
            return Err(RangeStepError::Overflow);
        }

        let offset = pow10(end_pad).saturating_mul(high.value);
        if idrs.end.saturating_add(offset) > MAX_U32_ID {
            return Err(RangeStepError::Overflow);
        }

        Ok(Self { idrs, high })
    }
}

impl RankRanges for AffixIdRangeStep {
    fn rank_ranges(&self) -> Vec<(u32, u32, u32)> {
        if self.high.value == 0 && self.high.pad == 0 {
            return self.idrs.rank_ranges();
        }

        let mut res = vec![];

        let mut start = self.idrs.start;
        let mut start_pad = self.idrs.pad;
        let mut bound = pow10(self.idrs.pad);
        let mut offset = bound.saturating_mul(self.high.value);

        while start <= self.idrs.end {
            while start >= bound {
                bound = bound.saturating_mul(10);
                offset = offset.saturating_mul(10);
                start_pad += 1;
            }

            let pad = start_pad + self.high.pad;

            if bound > self.idrs.end {
                bound = self.idrs.end + 1;
            }

            res.extend(
                IdRangeStep {
                    start: start + offset,
                    end: bound - 1 + offset,
                    step: self.idrs.step,
                    pad,
                }
                .rank_ranges(),
            );

            let remainder = (bound - 1 - start) % self.idrs.step;
            start = match bound.checked_add(self.idrs.step - remainder - 1) {
                Some(s) => s,
                None => break,
            };
        }

        res
    }

    fn start_rank(&self) -> u32 {
        let bound = pow10(self.idrs.pad);
        let offset = bound * self.high.value;

        padded_id_to_rank_exact(self.idrs.start + offset, self.idrs.pad + self.high.pad)
    }
}

#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub(crate) struct IdRangeOffset {
    pub(crate) value: u32,
    pub(crate) pad: u32,
}

impl IdRangeOffset {
    pub fn new(value: u32, pad: u32) -> Result<Self, RangeStepError> {
        if value > MAX_U32_ID || pad > MAX_U32_PAD {
            return Err(RangeStepError::Overflow);
        }

        Ok(Self { value, pad })
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
            // If we hit u32::MAX the jump pad won't be a power of 10 but we will never
            // use the value as id cannot be >= to it
            jump_pad = jump_pad.saturating_mul(10);
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
        let jump_rank = (self.rank - self.id).saturating_add(self.jump_pad);
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
/// The pad must be exact i.e the length of the displayed id must be equal to
/// the pad even if there are no zeroes at the front
/// # Examples
///
/// assert_eq!(padded_id_to_rank(423, 4), 1533);
fn padded_id_to_rank_exact(id: u32, pad: u32) -> u32 {
    let offset = rank_offset(pad);

    id + offset
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

fn pow10(n: u32) -> u32 {
    u32::saturating_pow(10, n)
}

fn rank_offset(n: u32) -> u32 {
    RANK_OFFSET_LOOKUP[n as usize]
}

const RANK_OFFSET_LOOKUP: [u32; 12] = [
    0,
    0,
    10,
    110,
    1110,
    11110,
    111110,
    1111110,
    11111110,
    111111110,
    1111111110,
    u32::MAX,
];
const MAX_U32_PAD: u32 = 10;
const MAX_U32_ID: u32 = u32::MAX - 1111111110 - 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rangestep_new() {
        assert!(IdRangeStep::new(0, 20, 1, 1).is_ok());
        assert!(IdRangeStep::new(20, 0, 0, 1).is_err());
        assert!(IdRangeStep::new(0, 4000000000, 1, 0).is_err());
        assert!(IdRangeStep::new(0, 20, 4000000000, 1).is_ok());
        assert!(IdRangeStep::new(0, 20, 1, 11).is_err());
    }
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
                pad: 1
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
                pad: 1
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

        assert_eq!(
            IdRangeStep {
                start: 0,
                end: MAX_U32_ID,
                step: 1000,
                pad: 1
            }
            .rank_ranges()
            .iter()
            .map(|(start, end, step)| (rank_to_string(*start), rank_to_string(*end), *step))
            .collect::<Vec<_>>(),
            vec![
                ("0".to_string(), "9".to_string(), 1000),
                ("1000".to_string(), "9999".to_string(), 1000),
                ("10000".to_string(), "99999".to_string(), 1000),
                ("100000".to_string(), "999999".to_string(), 1000),
                ("1000000".to_string(), "9999999".to_string(), 1000),
                ("10000000".to_string(), "99999999".to_string(), 1000),
                ("100000000".to_string(), "999999999".to_string(), 1000),
                ("1000000000".to_string(), format!("{MAX_U32_ID}"), 1000),
            ]
        );
    }

    use std::num::ParseIntError;
    fn rank_of_string(s: &str) -> Result<u32, ParseIntError> {
        Ok(padded_id_to_rank_exact(s.parse::<u32>()?, s.len() as u32))
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
        assert_eq!(
            rank_of_string(&format!("{MAX_U32_ID}")).unwrap(),
            u32::MAX - 1
        );
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
        assert_eq!(format!("{MAX_U32_ID}"), rank_to_string(u32::MAX - 1));
    }

    // Displays a rank as a zero-padded string
    // fn rank_to_string(rank: u32) -> String {
    //     CachedTranslation::new(rank).to_string()
    // }

    #[test]
    fn test_affix_rank_ranges_default() {
        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 8,
                end: 12,
                step: 1,
                pad: 1,
            },
            Some(IdRangeOffset { value: 0, pad: 0 }),
            Some(IdRangeOffset { value: 0, pad: 0 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![
                ("8".to_string(), "9".to_string(), 1u32),
                ("10".to_string(), "12".to_string(), 1u32)
            ]
        );
    }

    #[test]
    fn test_affix_rank_ranges_low() {
        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 8,
                end: 12,
                step: 1,
                pad: 1,
            },
            Some(IdRangeOffset { value: 50, pad: 2 }),
            Some(IdRangeOffset { value: 0, pad: 0 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![
                ("850".to_string(), "999".to_string(), 100u32),
                ("1050".to_string(), "1250".to_string(), 100u32),
            ]
        );

        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 8,
                end: 12,
                step: 1,
                pad: 1,
            },
            Some(IdRangeOffset { value: 5, pad: 2 }),
            Some(IdRangeOffset { value: 0, pad: 0 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![
                ("805".to_string(), "999".to_string(), 100u32),
                ("1005".to_string(), "1205".to_string(), 100u32),
            ]
        );

        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 8,
                end: 12,
                step: 1,
                pad: 2,
            },
            Some(IdRangeOffset { value: 5, pad: 2 }),
            Some(IdRangeOffset { value: 0, pad: 0 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![("0805".to_string(), "1205".to_string(), 100u32),]
        );
    }

    #[test]
    fn test_affix_rank_ranges_high() {
        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 8,
                end: 12,
                step: 1,
                pad: 1,
            },
            Some(IdRangeOffset { value: 0, pad: 0 }),
            Some(IdRangeOffset { value: 50, pad: 2 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![
                ("508".to_string(), "509".to_string(), 1u32),
                ("5010".to_string(), "5012".to_string(), 1u32),
            ]
        );

        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 8,
                end: 12,
                step: 1,
                pad: 2,
            },
            Some(IdRangeOffset { value: 0, pad: 0 }),
            Some(IdRangeOffset { value: 5, pad: 2 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![("0508".to_string(), "0512".to_string(), 1u32),]
        );

        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 1,
                end: 100,
                step: 1,
                pad: 1,
            },
            Some(IdRangeOffset { value: 0, pad: 0 }),
            Some(IdRangeOffset { value: 5, pad: 1 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![
                ("51".to_string(), "59".to_string(), 1u32),
                ("510".to_string(), "599".to_string(), 1u32),
                ("5100".to_string(), "5100".to_string(), 1u32),
            ]
        );
    }

    #[test]
    fn test_affix_rank_ranges_low_high() {
        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 0,
                end: 5,
                step: 1,
                pad: 1,
            },
            Some(IdRangeOffset { value: 50, pad: 2 }),
            Some(IdRangeOffset { value: 50, pad: 2 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![("50050".to_string(), "50550".to_string(), 100u32),]
        );

        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 8,
                end: 12,
                step: 1,
                pad: 1,
            },
            Some(IdRangeOffset { value: 50, pad: 2 }),
            Some(IdRangeOffset { value: 50, pad: 2 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![
                ("50850".to_string(), "50999".to_string(), 100u32),
                ("501050".to_string(), "501250".to_string(), 100u32),
            ]
        );

        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 8,
                end: 12,
                step: 1,
                pad: 1,
            },
            Some(IdRangeOffset { value: 5, pad: 2 }),
            Some(IdRangeOffset { value: 5, pad: 2 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![
                ("05805".to_string(), "05999".to_string(), 100u32),
                ("051005".to_string(), "051205".to_string(), 100u32),
            ]
        );

        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 8,
                end: 12,
                step: 1,
                pad: 2,
            },
            Some(IdRangeOffset { value: 5, pad: 2 }),
            Some(IdRangeOffset { value: 5, pad: 2 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![("050805".to_string(), "051205".to_string(), 100u32),]
        );
    }

    #[test]
    fn test_affix_rank_ranges_steps() {
        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 0,
                end: 10000,
                step: 2500,
                pad: 2,
            },
            Some(IdRangeOffset { value: 0, pad: 0 }),
            Some(IdRangeOffset { value: 0, pad: 0 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![
                ("00".to_string(), "99".to_string(), 2500u32),
                ("2500".to_string(), "9999".to_string(), 2500u32),
                ("10000".to_string(), "10000".to_string(), 2500u32),
            ]
        );

        let affix = AffixIdRangeStep::new(
            IdRangeStep {
                start: 0,
                end: 10000,
                step: 2500,
                pad: 2,
            },
            Some(IdRangeOffset { value: 5, pad: 2 }),
            Some(IdRangeOffset { value: 5, pad: 2 }),
        )
        .unwrap();

        assert_eq!(
            affix
                .rank_ranges()
                .iter()
                .map(|(s, e, st)| (rank_to_string(*s), rank_to_string(*e), *st))
                .collect::<Vec<_>>(),
            vec![
                ("050005".to_string(), "059999".to_string(), 250000u32),
                ("05250005".to_string(), "05999999".to_string(), 250000u32),
                ("051000005".to_string(), "051000005".to_string(), 250000u32),
            ]
        );
    }
}
