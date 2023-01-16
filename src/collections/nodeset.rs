use super::parsers;
use crate::idrange::rank_to_string;
use crate::idrange::CachedTranslation;
use crate::idrange::IdRange;
use crate::{IdSet, IdSetIter};
use itertools::Itertools;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NodeSet<T = crate::IdRangeList> {
    pub(crate) dimnames: HashMap<NodeSetDimensions, IdSetKind<T>>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum IdSetKind<T> {
    None,
    Single(T),
    Multiple(IdSet<T>),
}

impl<T> Default for NodeSet<T> {
    fn default() -> Self {
        Self {
            dimnames: HashMap::new(),
        }
    }
}

pub struct NodeSetIter<'a, T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    dim_iter:
        std::iter::Peekable<std::collections::hash_map::Iter<'a, NodeSetDimensions, IdSetKind<T>>>,
    set_iter: IdSetIterKind<'a, T>,
    cache: Option<CachedTranslation>,
}

enum IdSetIterKind<'a, T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    None,
    Single(T::SelfIter<'a>),
    Multiple(IdSetIter<'a, T>),
}

impl<'b, T> NodeSetIter<'b, T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    fn new(dims: &'b HashMap<NodeSetDimensions, IdSetKind<T>>) -> Self {
        let mut it = Self {
            dim_iter: dims.iter().peekable(),
            set_iter: IdSetIterKind::None,
            cache: None,
        };
        it.init_dims();
        it
    }

    fn next_dims(&mut self) {
        self.dim_iter.next();
        self.init_dims()
    }

    fn init_dims(&mut self) {
        self.set_iter = self
            .dim_iter
            .peek()
            .map(|s| match s.1 {
                IdSetKind::None => IdSetIterKind::None,
                IdSetKind::Single(s) => IdSetIterKind::Single(s.iter()),
                IdSetKind::Multiple(s) => IdSetIterKind::Multiple(s.iter()),
            })
            .unwrap_or(IdSetIterKind::None);
    }
}

impl<'b, T> Iterator for NodeSetIter<'b, T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let dimnames = &self.dim_iter.peek()?.0.dimnames;

            match &mut self.set_iter {
                IdSetIterKind::None => {
                    self.next_dims();
                    return Some(dimnames[0].clone());
                }
                IdSetIterKind::Single(set_iter) => {
                    if let Some(coord) = set_iter.next() {
                        let cache = self
                            .cache
                            .get_or_insert_with(|| CachedTranslation::new(*coord));

                        return Some(format!("{}{}", dimnames[0], cache.interpolate(*coord)));
                    } else {
                        self.next_dims();
                    }
                }
                IdSetIterKind::Multiple(set_iter) => {
                    if let Some(coords) = set_iter.next() {
                        return Some(
                            dimnames
                                .iter()
                                .zip(coords.iter())
                                .map(|(a, b)| format!("{}{}", a, rank_to_string(*b)))
                                .join(""),
                        );
                    } else {
                        self.next_dims();
                    }
                }
            }
        }
    }
}

impl<T> NodeSet<T>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    pub fn new() -> Self {
        NodeSet {
            dimnames: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.dimnames
            .values()
            .map(|set| match set {
                IdSetKind::None => 1,
                IdSetKind::Single(set) => set.len(),
                IdSetKind::Multiple(set) => set.len(),
            })
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.dimnames.is_empty()
    }

    pub fn iter(&self) -> NodeSetIter<'_, T> {
        NodeSetIter::new(&self.dimnames)
    }

    pub fn fold(&mut self) -> &mut Self {
        self.dimnames.values_mut().for_each(|s| match s {
            IdSetKind::None => {}
            IdSetKind::Single(set) => {
                set.sort();
            }
            IdSetKind::Multiple(set) => {
                set.fold();
            }
        });

        self
    }

    pub fn extend(&mut self, other: &Self) {
        for (dimname, oset) in other.dimnames.iter() {
            match self.dimnames.get_mut(dimname) {
                None => {
                    self.dimnames.insert(dimname.clone(), oset.clone());
                }
                Some(set) => match set {
                    IdSetKind::None => {}
                    IdSetKind::Single(set) => {
                        let  IdSetKind::Single(oset) = oset else {
                                panic!("Mismatched set kinds");
                            };
                        set.push(oset);
                    }
                    IdSetKind::Multiple(set) => {
                        let  IdSetKind::Multiple(oset) = oset else {
                                panic!("Mismatched set kinds");
                            };
                        set.extend(oset);
                    }
                },
            };
        }
    }

    pub fn difference(&self, other: &Self) -> Self {
        let mut dimnames = HashMap::<NodeSetDimensions, IdSetKind<T>>::new();
        for (dimname, set) in self.dimnames.iter() {
            if let Some(oset) = other.dimnames.get(dimname) {
                match (set, oset) {
                    (IdSetKind::None, IdSetKind::None) => continue,
                    (IdSetKind::Single(set), IdSetKind::Single(oset)) => {
                        let result = T::from_sorted(set.difference(oset));
                        if !result.is_empty() {
                            dimnames.insert(dimname.clone(), IdSetKind::Single(result));
                        }
                    }
                    (IdSetKind::Multiple(set), IdSetKind::Multiple(oset)) => {
                        if let Some(nset) = set.difference(oset) {
                            dimnames.insert(dimname.clone(), IdSetKind::Multiple(nset));
                        }
                    }
                    _ => {
                        panic!("Mismatched set kinds");
                    }
                }
            } else {
                dimnames.insert(dimname.clone(), set.clone());
            }
        }
        NodeSet { dimnames }
    }

    pub fn intersection(&self, other: &Self) -> Self {
        let mut dimnames = HashMap::<NodeSetDimensions, IdSetKind<T>>::new();
        for (dimname, set) in self.dimnames.iter() {
            if let Some(oset) = other.dimnames.get(dimname) {
                match (set, oset) {
                    (IdSetKind::None, IdSetKind::None) => continue,
                    (_, IdSetKind::None) => {
                        dimnames.insert(dimname.clone(), set.clone());
                    }
                    (IdSetKind::Single(set), IdSetKind::Single(oset)) => {
                        let result = T::from_sorted(set.intersection(oset));
                        if !result.is_empty() {
                            dimnames.insert(dimname.clone(), IdSetKind::Single(result));
                        }
                    }
                    (IdSetKind::Multiple(set), IdSetKind::Multiple(oset)) => {
                        if let Some(nset) = set.intersection(oset) {
                            dimnames.insert(dimname.clone(), IdSetKind::Multiple(nset));
                        }
                    }
                    _ => {
                        panic!("Mismatched set kinds");
                    }
                }
            }
        }

        NodeSet { dimnames }
    }

    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut dimnames = HashMap::<NodeSetDimensions, IdSetKind<T>>::new();
        for (dimname, set) in self.dimnames.iter() {
            if let Some(oset) = other.dimnames.get(dimname) {
                match (set, oset) {
                    (IdSetKind::None, IdSetKind::None) => continue,
                    (IdSetKind::Single(set), IdSetKind::Single(oset)) => {
                        let result = T::from_sorted(set.symmetric_difference(oset));
                        if !result.is_empty() {
                            dimnames.insert(dimname.clone(), IdSetKind::Single(result));
                        }
                    }
                    (IdSetKind::Multiple(set), IdSetKind::Multiple(oset)) => {
                        if let Some(nset) = set.symmetric_difference(oset) {
                            dimnames.insert(dimname.clone(), IdSetKind::Multiple(nset));
                        }
                    }
                    _ => {
                        panic!("Mismatched set kinds");
                    }
                }
            } else {
                dimnames.insert(dimname.clone(), set.clone());
            }
        }
        for (dimname, set) in other.dimnames.iter() {
            if !self.dimnames.contains_key(dimname) {
                dimnames.insert(dimname.clone(), set.clone());
            }
        }
        NodeSet { dimnames }
    }
}

use parsers::CustomError;

impl<T> std::str::FromStr for NodeSet<T>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    type Err = NodeSetParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parsers::full_expr::<T>(s)
            .map(|r| r.1)
            .map_err(|e| match e {
                nom::Err::Error(e) => NodeSetParseError::from(e),
                nom::Err::Failure(e) => NodeSetParseError::from(e),
                _ => panic!("unreachable"),
            })
    }
}

impl From<CustomError<&str>> for NodeSetParseError {
    fn from(e: CustomError<&str>) -> Self {
        match e {
            CustomError::NodeSetError(e) => e,
            CustomError::Nom(e, _) => NodeSetParseError::Generic(e.to_string()),
        }
    }
}

/// List of names for each dimension of a NodeSet along with an optional suffix
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct NodeSetDimensions {
    dimnames: Vec<String>,
    has_suffix: bool,
}

impl NodeSetDimensions {
    /*     fn is_unique(&self) -> bool {
        return self.dimnames.len() == 1 && self.has_suffix
    } */
    pub fn new() -> NodeSetDimensions {
        NodeSetDimensions {
            dimnames: Vec::<String>::new(),
            has_suffix: false,
        }
    }
    /*todo: better manage has_suffix, check consistency (single suffix)*/
    pub fn push(&mut self, d: &str, has_suffix: bool) {
        self.dimnames.push(d.into());
        self.has_suffix = has_suffix;
    }
}

impl<T> fmt::Display for NodeSet<T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut first = true;

        for (dim, set) in &self.dimnames {
            if !first {
                write!(f, ",")?;
            }
            match set {
                IdSetKind::None => {
                    write!(f, "{}", dim.dimnames[0])?;
                }
                IdSetKind::Single(set) => {
                    write!(f, "{}{}", dim.dimnames[0], set)?;
                }
                IdSetKind::Multiple(set) => {
                    set.fmt_dims(f, &dim.dimnames)
                        .expect("failed to format string");
                }
            }

            first = false;
        }
        Ok(())
    }
}

use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum NodeSetParseError {
    #[error("invalid integer")]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error("value out of range")]
    OverFlow(#[from] std::num::TryFromIntError),
    #[error("inverted range '{0}'")]
    Reverse(String),
    #[error("unable to parse '{0}'")]
    Generic(String),
    #[error("mismatched padding: '{0}'")]
    Padding(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idrange::IdRangeList;

    #[test]
    fn test_nodeset_empty() {
        let id1: NodeSet<IdRangeList> = "".parse().unwrap();
        assert!(id1.is_empty());

        let id1: NodeSet<IdRangeList> = "x[1-2],b".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "x[3-4],c".parse().unwrap();

        assert!(id1.intersection(&id2).is_empty());
    }

    #[test]
    fn test_nodeset_parse() {
        let id1: NodeSet<IdRangeList> = "x[1-10/2,5]y[1-7]z3,x[1-10/2,5]y[1-7]z2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "x[2-5]y[7]z[2,3]".parse().unwrap();

        assert_eq!(
            id1.to_string(),
            "x[1,3,5,7,9]y[1-7]z[3],x[1,3,5,7,9]y[1-7]z[2]"
        );
        assert_eq!(id2.to_string(), "x[2-5]y[7]z[2-3]");
        assert_eq!(
            id1.intersection(&id2).to_string(),
            "x[3,5]y[7]z[3],x[3,5]y[7]z[2]"
        );
    }

    #[test]
    fn test_nodeset_intersect() {
        let id1: NodeSet<IdRangeList> = "x[1-10/2,5]y[1-7]z3,x[1-10/2,5]y[1-7]z2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "x[2-5]y[7]z[2,3]".parse().unwrap();

        assert_eq!(
            id1.intersection(&id2).fold().to_string(),
            "x[3,5]y[7]z[2-3]"
        );
    }

    #[test]
    fn test_nodeset_len() {
        let id1: NodeSet<IdRangeList> = "a b".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "a[0-9]b".parse().unwrap();
        let id3: NodeSet<IdRangeList> = "a[0-9]b[0-8]".parse().unwrap();
        let id4: NodeSet<IdRangeList> = "a[0-10000] b[0-100]".parse().unwrap();

        assert_eq!(id1.len(), 2);
        assert_eq!(id2.len(), 10);
        assert_eq!(id3.len(), 90);
        assert_eq!(id4.len(), 10102);
    }

    #[test]
    fn test_nodeset_fold() {
        let mut id1: NodeSet<IdRangeList> =
            "a[1-10/2,5]b[1-7]c3,a[1-10/2,5]b[1-7]c2".parse().unwrap();
        let mut id2: NodeSet<IdRangeList> = "a[0-10]b[0-10],a[0-20]b[0-10]".parse().unwrap();
        let mut id3: NodeSet<IdRangeList> = "x[0-10]y[0-10],x[8-18]y[8-18],x[11-18]y[0-7]"
            .parse()
            .unwrap();

        id1.fold();
        id2.fold();
        id3.fold();

        assert_eq!(id1.to_string(), "a[1,3,5,7,9]b[1-7]c[2-3]");
        assert_eq!(id2.to_string(), "a[0-20]b[0-10]");
        assert_eq!(id3.to_string(), "x[0-7]y[0-10],x[8-18]y[0-18]");
    }

    #[test]
    fn test_nodeset_iter() {
        let id1: NodeSet<IdRangeList> = "a[1-2]b[1-2]".parse().unwrap();

        assert_eq!(
            id1.iter().collect::<Vec<_>>(),
            vec!["a1b1", "a1b2", "a2b1", "a2b2",]
        );
    }
}
