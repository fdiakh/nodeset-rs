use super::parsers::CustomError;
use super::parsers::Parser;
use crate::idrange::rank_to_string;
use crate::idrange::CachedTranslation;
use crate::idrange::IdRange;
use crate::Resolver;
use crate::{IdSet, IdSetIter};
use itertools::Itertools;
use std::collections::HashMap;
use std::fmt;

/// NodeSet structure stores a nodeset definition
///
/// As its internal in memory structure it uses [IdRangeList](crate::IdRangeList) (backed by a Vec structure)
/// by default but it can also use [IdRangeTree](crate::IdRangeTree) (backed by a BTreeSet structure).
/// `IdRangeList` seems to be more efficient this is why it is provided by default.
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
                        let has_suffix = self.dim_iter.peek()?.0.has_suffix;
                        if has_suffix && dimnames.len() >1 {
                            return Some(format!("{}{}{}", dimnames[0], cache.interpolate(*coord), dimnames[1]));
                        } else {
                            return Some(format!("{}{}", dimnames[0], cache.interpolate(*coord)));
                        }
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
    /// Creates a new empty NodeSet.
    pub fn new() -> Self {
        NodeSet {
            dimnames: HashMap::new(),
        }
    }

    /// return NodeSet's number of elements.
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

    /// returns true if NodeSet has no element.
    pub fn is_empty(&self) -> bool {
        self.dimnames.is_empty()
    }

    /// Returns a NodeSetIter iterator on the NodeSet
    pub fn iter(&self) -> NodeSetIter<'_, T> {
        NodeSetIter::new(&self.dimnames)
    }

    /// Folds the NodeSet:
    ///  * `a[1-10,12,13,14]` -> `a[1-10,12-14]`
    ///  * `a[1,2,3,4]b[2,3,4,5,6]` -> `a[1-4]b[2-6]`
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

    /// Realizes the union between NodeSets self and other that
    /// should be of the same IdSetKind. This function will panic
    /// if the NodeSets are of different IdSetKind.
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

    /// Computes the difference between the two NodeSets and
    /// returns a new NodeSet.
    /// This function will panic if NodeSets are of different IdSetKind
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

    /// Computes the intersection between the two NodeSets (keeping
    /// nodes that are in both NodeSets) and returns a new NodeSet.
    /// This function will panic if NodeSets are of different IdSetKind
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

    /// Computes a symmetric difference on the two NodeSets.
    /// This function will panic if NodeSets are of different IdSetKind
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

impl<T> std::str::FromStr for NodeSet<T>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    type Err = NodeSetParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let resolver = Resolver::get_global();
        Parser::with_resolver(resolver.as_ref(), None).parse::<T>(s)
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
                    if dim.has_suffix && dim.dimnames.len() > 1 {
                        write!(f, "{}{}{}", dim.dimnames[0], set, dim.dimnames[1])?;
                    } else {
                        write!(f, "{}{}", dim.dimnames[0], set)?;
                    }
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


/// Possible errors that may happen when parsing nodesets and `ns` program's configuration files.
#[derive(thiserror::Error, Debug)]
pub enum NodeSetParseError {
    /// Integer value is not correct.
    #[error("invalid integer")]
    ParseIntError(#[from] std::num::ParseIntError),

    /// Static configuration file is not correct.
    #[error("invalid static configuration file")]
    StaticConfiguration(#[from] serde_yaml::Error),

    /// Dynamic configuration file is not correct.
    #[error("invalid configuration file")]
    DynamicConfiguration(#[from] serde_ini::de::Error),

    /// Value is out of range (should be within `u32` limits).
    #[error("value out of range")]
    OverFlow(#[from] std::num::TryFromIntError),

    /// An error occurred while executing an external command as specified in the dynamic configuration file.
    #[error("external command execution failed")]
    Command(#[from] std::io::Error),

    /// Range is inverted (ie `[9-2]`).
    #[error("inverted range '{0}'")]
    Reverse(String),

    /// Parsing failed.
    #[error("unable to parse '{0}'")]
    Generic(String),

    /// Padding does not correspond (ie `[01-003]`).
    #[error("mismatched padding: '{0}'")]
    Padding(String),

    /// Group source may not be defined
    #[error("Unknown group source: '{0}'")]
    Source(String),

    /// Group is unknown within the specified source.
    #[error("Unknown group: '{1}' in source: '{0}'")]
    Group(String, String),
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
    fn test_nodeset_parse_with_suffix() {
        let id1: NodeSet<IdRangeList> = "a[1-3]_e".parse().unwrap();

        assert_eq!(id1.to_string(), "a[1-3]_e");
    }

    #[test]
    fn test_nodeset_intersect() {
        let id1: NodeSet<IdRangeList> = "x[1-10/2,5]y[1-7]z3,x[1-10/2,5]y[1-7]z2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "x[2-5]y[7]z[2,3]".parse().unwrap();

        assert_eq!(
            id1.intersection(&id2).fold().to_string(),
            "x[3,5]y[7]z[2-3]"
        );

        let id1: NodeSet<IdRangeList> = "a1 a2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "b1 b2".parse().unwrap();

        assert!(id1.intersection(&id2).is_empty(),);
    }

    #[test]
    fn test_nodeset_symmetric_diff() {
        let id1: NodeSet<IdRangeList> = "x[1-10/2,5]y[1-7]z3,x[1-10/2,5]y[1-7]z2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "x[2-5]y[7]z[2,3]".parse().unwrap();

        assert_eq!(
            id1.symmetric_difference(&id2).fold().to_string(),
            "x[1,7,9]y[1-7]z[2-3],x[2,4]y[7]z[2-3],x[3,5]y[1-6]z[2-3]"
        );

        let id1: NodeSet<IdRangeList> = "a1 b1 a2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "b1 a2 a3".parse().unwrap();

        assert_eq!(id1.symmetric_difference(&id2).to_string(), "a[1,3]");
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

    #[test]
    fn test_nodeset_padded() {
        assert_eq!(
            "a01 a02 a9 a0 a00 a1 a09 a10"
                .parse::<NodeSet<IdRangeList>>()
                .unwrap()
                .to_string(),
            "a[0-1,9,00-02,09-10]"
        );

        assert_eq!(
            "n[001-999] n[1000]"
                .parse::<NodeSet<IdRangeList>>()
                .unwrap()
                .to_string(),
            "n[001-999,1000]"
        );

        assert_eq!(
            "n[000-999] n[1000] n1001 n0 n1"
                .parse::<NodeSet<IdRangeList>>()
                .unwrap()
                .to_string(),
            "n[0-1,000-999,1000-1001]"
        );

        assert_eq!(
            "n[0000-1000] n[1000-10000] n1001 n1 n00"
                .parse::<NodeSet<IdRangeList>>()
                .unwrap()
                .to_string(),
            "n[1,00,0000-9999,10000]"
        );
    }
}
