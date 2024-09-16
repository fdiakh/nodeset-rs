use super::parsers::CustomError;
use super::parsers::Parser;
use crate::idrange::CachedTranslation;
use crate::idrange::IdRange;
use crate::idrange::RangeStepError;
use crate::Resolver;
use crate::{IdSet, IdSetIter};
use std::collections::BTreeMap;
use std::fmt;

/// An unordered collection of nodes indexed in one or more dimensions.
///
/// Two implementations are provided:
/// * `NodeSet<IdRangeList>` which stores node indices in Vecs
/// * `NodeSet<IdRangeTree>` which stores node indices in BTrees
///
/// By default `IdRangeList` are used as they are faster to build for one shot
/// operations which are the most common, especially when using the CLI.
/// However, if many updates are performed on a large NodeSet `IdRangeTree` may
/// more efficient especially for one-dimensional NodeSets.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NodeSet<T = crate::IdRangeList> {
    pub(crate) bases: BTreeMap<NodeSetDimensions, IdSetKind<T>>,
    lazy: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum IdSetKind<T> {
    None,
    Single(T),
    Multiple(IdSet<T>),
}

impl NodeSet<crate::IdRangeList> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T> Default for NodeSet<T> {
    fn default() -> Self {
        Self {
            bases: BTreeMap::new(),
            lazy: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeSetIter<'a, T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    dim_iter:
        std::iter::Peekable<std::collections::btree_map::Iter<'a, NodeSetDimensions, IdSetKind<T>>>,
    set_iter: IdSetIterKind<'a, T>,
    cache: Option<CachedTranslation>,
}

#[derive(Debug, Clone)]
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
    fn new(dims: &'b BTreeMap<NodeSetDimensions, IdSetKind<T>>) -> Self {
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
        self.cache = None;
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
            let dim = self.dim_iter.peek()?.0;

            match &mut self.set_iter {
                IdSetIterKind::None => {
                    self.next_dims();
                    return Some(dim.dimnames[0].clone());
                }
                IdSetIterKind::Single(set_iter) => {
                    if let Some(coord) = set_iter.next() {
                        let cache = self
                            .cache
                            .as_ref()
                            .map(|c| c.interpolate(coord))
                            .unwrap_or_else(|| CachedTranslation::new(coord));

                        let mut res = String::with_capacity(
                            dim.dimnames.iter().map(|s| s.len()).sum::<usize>()
                                + cache.padding() as usize
                                + 1,
                        );

                        dim.fmt_ranges(&mut res, [&cache])
                            .expect("string format should succeed");

                        self.cache = Some(cache);
                        return Some(res);
                    } else {
                        self.next_dims();
                    }
                }
                IdSetIterKind::Multiple(set_iter) => {
                    if let Some(coords) = set_iter.next() {
                        let mut res = String::new();
                        dim.fmt_ranges(&mut res, coords.iter().map(CachedTranslation::new))
                            .expect("string format should succeed");
                        return Some(res);
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
    /// Returns the number of elements in the set
    pub fn len(&self) -> usize {
        self.bases
            .values()
            .map(|set| match set {
                IdSetKind::None => 1,
                IdSetKind::Single(set) => set.len(),
                IdSetKind::Multiple(set) => set.len(),
            })
            .sum()
    }

    /// Returns true if the set contains no element
    pub fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    /// Returns an iterator over all elements of the set
    pub fn iter(&self) -> NodeSetIter<'_, T> {
        NodeSetIter::new(&self.bases)
    }

    /// Folds and deduplicates the internal representation of the set
    ///
    /// This method is automatically called after each operation unless the
    /// NodeSet is in lazy mode. When in lazy mode, the NodeSet must be folded
    /// again before calling len() otherwise nodes may be counted multiple
    /// times.
    pub(crate) fn fold(&mut self) -> &mut Self {
        self.bases.values_mut().for_each(|s| match s {
            IdSetKind::None => {}
            IdSetKind::Single(set) => {
                set.sort();
            }
            IdSetKind::Multiple(set) => {
                set.fold();
            }
        });

        self.lazy = false;
        self
    }

    /// Adds elements from `other` to `self`
    pub(crate) fn extend_from_nodeset(&mut self, other: &Self) {
        for (dimname, oset) in other.bases.iter() {
            match self.bases.get_mut(dimname) {
                None => {
                    self.bases.insert(dimname.clone(), oset.clone());
                }
                Some(set) => match set {
                    IdSetKind::None => {
                        let IdSetKind::None = oset else {
                            panic!("Mismatched set kinds");
                        };
                    }
                    IdSetKind::Single(set) => {
                        let IdSetKind::Single(oset) = oset else {
                            panic!("Mismatched set kinds");
                        };
                        set.push(oset);
                    }
                    IdSetKind::Multiple(set) => {
                        let IdSetKind::Multiple(oset) = oset else {
                            panic!("Mismatched set kinds");
                        };
                        set.extend(oset);
                    }
                },
            };
        }
    }

    /// Returns a new set containing elements found in `self` and `other`
    pub fn union(&self, other: &Self) -> Self {
        let mut res = self.clone();

        res.extend_from_nodeset(other);
        if !self.lazy {
            res.fold();
        }

        res
    }

    /// Returns a new set containing elements found in `self` but not in `other`
    pub fn difference(&self, other: &Self) -> Self {
        let mut dimnames = BTreeMap::<NodeSetDimensions, IdSetKind<T>>::new();
        for (dimname, set) in self.bases.iter() {
            if let Some(oset) = other.bases.get(dimname) {
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

        NodeSet::from_dims(dimnames, self.lazy)
    }

    /// Returns a new set containing elements that are in both `self` and `other`
    pub fn intersection(&self, other: &Self) -> Self {
        let mut dimnames = BTreeMap::<NodeSetDimensions, IdSetKind<T>>::new();
        for (dimname, set) in self.bases.iter() {
            if let Some(oset) = other.bases.get(dimname) {
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

        NodeSet::from_dims(dimnames, self.lazy)
    }

    /// Returns a new set containing the elements found in either `self` or `other` but not in both
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut dimnames = BTreeMap::<NodeSetDimensions, IdSetKind<T>>::new();
        for (dimname, set) in self.bases.iter() {
            if let Some(oset) = other.bases.get(dimname) {
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
        for (dimname, set) in other.bases.iter() {
            if !self.bases.contains_key(dimname) {
                dimnames.insert(dimname.clone(), set.clone());
            }
        }
        NodeSet::from_dims(dimnames, self.lazy)
    }

    /// Create a NodeSet from a mapping of NodeSetDimensions to IdSets
    fn from_dims(dimnames: BTreeMap<NodeSetDimensions, IdSetKind<T>>, lazy: bool) -> Self {
        let mut res = NodeSet {
            bases: dimnames,
            lazy,
        };

        if !lazy {
            res.fold();
        }

        res
    }

    /// Create a new lazy NodeSet which does not automatically folds after each
    /// operation
    pub(crate) fn lazy() -> Self {
        NodeSet {
            bases: BTreeMap::new(),
            lazy: true,
        }
    }
}

impl<T> std::str::FromStr for NodeSet<T>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    type Err = NodeSetParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let resolver = Resolver::get_global();
        Parser::with_resolver(resolver, None).parse::<T>(s)
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
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Default, Debug)]
pub(crate) struct NodeSetDimensions {
    /// The names associated with each dimension of a nodeset
    dimnames: Vec<String>,
    /// If true, the last name is a suffix meaning that there is one more name
    /// than ranges
    has_suffix: bool,
}

impl NodeSetDimensions {
    pub(crate) fn new() -> NodeSetDimensions {
        Default::default()
    }

    pub(crate) fn push(&mut self, d: &str) {
        assert!(
            !self.has_suffix,
            "Cannot add a dimension name after a suffix has been added"
        );
        self.dimnames.push(d.into());
    }

    pub(crate) fn push_suffix(&mut self, d: &str) {
        self.has_suffix = true;
        self.dimnames.push(d.into());
    }

    pub(crate) fn fmt_ranges<T>(
        &self,
        f: &mut dyn fmt::Write,
        ranges: impl IntoIterator<Item = T>,
    ) -> fmt::Result
    where
        T: fmt::Display,
    {
        let mut dimnames = self.dimnames.iter();
        for r in ranges.into_iter() {
            f.write_str(
                dimnames
                    .next()
                    .expect("should be at least as many names as ranges"),
            )?;
            if self.is_rangeset() {
                write!(f, "{:#}", r)?;
            } else {
                write!(f, "{}", r)?;
            }
        }

        if let Some(suffix) = dimnames.next() {
            f.write_str(suffix)?;
        }

        Ok(())
    }

    fn is_rangeset(&self) -> bool {
        self.dimnames.len() == 1 && self.dimnames[0].is_empty()
    }
}

impl<T> fmt::Display for NodeSet<T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut first = true;

        for (dim, set) in &self.bases {
            if !first {
                f.write_str(",")?;
            }
            match set {
                IdSetKind::None => {
                    write!(f, "{}", dim.dimnames[0])?;
                }
                IdSetKind::Single(set) => {
                    dim.fmt_ranges(f, [set])?;
                }
                IdSetKind::Multiple(set) => {
                    set.fmt_dims(f, dim).expect("failed to format string");
                }
            }

            first = false;
        }
        Ok(())
    }
}

/// Errors that may happen when parsing configuration files
#[derive(thiserror::Error, Debug)]
pub enum ConfigurationError {
    /// A YAML configuration file cannot be parsed
    #[error("invalid yaml file")]
    InvalidYamlFile(#[from] serde_yaml::Error),

    /// An INI configuration file cannot be parsed
    #[error("invalid ini file")]
    InvalidIniFile(#[from] ini::Error),

    /// A property is missing in the configuration file
    #[error("missing ini property: {0}")]
    MissingProperty(String),

    /// An unexpected property was found in the configuration file
    #[error("unexpected ini property: {0}")]
    UnexpectedProperty(String),
}

/// Errors that may happen when parsing nodesets
#[derive(thiserror::Error, Debug)]
pub enum NodeSetParseError {
    /// An error occurred while parsing an integer.
    #[error("invalid range")]
    RangeError(#[from] RangeStepError),

    /// An error occurred while parsing an integer.
    #[error("invalid integer")]
    ParseIntError(#[from] std::num::ParseIntError),

    /// An index is out of range.
    #[error("value out of range")]
    OverFlow(#[from] std::num::TryFromIntError),

    /// An error occurred while executing an external command as specified in the dynamic configuration file.
    #[error("external command execution failed")]
    Command(#[from] std::io::Error),

    /// A parsing error which does not fit in any other category.
    #[error("unable to parse '{0}'")]
    Generic(String),

    /// Padding sizes at both ends of a range do not match (ie `[01-003]`).
    #[error("mismatched padding: '{0}'")]
    Padding(String),

    /// A reference was made to a group source that does not exist.
    #[error("Unknown group source: '{0}'")]
    Source(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idrange::IdRangeList;

    fn parse_to_fold(ns: &str) -> Result<String, NodeSetParseError> {
        ns.parse::<NodeSet<IdRangeList>>().map(|ns| ns.to_string())
    }

    fn parse_to_vec(ns: &str) -> Result<Vec<String>, NodeSetParseError> {
        ns.parse::<NodeSet<IdRangeList>>().map(|ns| {
            let mut r: Vec<_> = ns.iter().collect();
            r.sort();
            r
        })
    }

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
        let id2: NodeSet<IdRangeList> = "x[2-5]y7z[2,3]".parse().unwrap();

        assert_eq!(id1.to_string(), "x[1,3,5,7,9]y[1-7]z[2-3]");
        assert_eq!(id2.to_string(), "x[2-5]y7z[2-3]");
        assert_eq!(id1.intersection(&id2).to_string(), "x[3,5]y7z[2-3]");
    }

    #[test]
    fn test_rangeset_parse() {
        let id1: NodeSet<IdRangeList> = "12,3".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "0-10/2,3".parse().unwrap();

        assert_eq!(id1.to_string(), "3,12");
        assert_eq!(id2.to_string(), "0,2-4,6,8,10");
        assert_eq!(id1.intersection(&id2).to_string(), "3");

        assert_eq!(parse_to_vec("[1,2]").unwrap(), vec!["1", "2"]);
    }

    #[test]
    fn test_rangeset_parse_affix() {
        assert_eq!(parse_to_vec("50[01,20]").unwrap(), vec!["5001", "5020"]);
        assert_eq!(parse_to_vec("[01,2]50").unwrap(), vec!["0150", "250"]);
        assert_eq!(
            parse_to_vec("05[1,10]05").unwrap(),
            vec!["051005", "05105",]
        );
        assert!(parse_to_vec("05[1,10]05[0]").is_err());
        assert!(parse_to_vec("[1][0]").is_err());
    }

    #[test]
    fn test_rangeset_nodeset_parse() {
        assert_eq!(parse_to_vec("1,2a").unwrap(), vec!["1", "2a"]);

        assert_eq!(parse_to_vec("1-2a").unwrap(), vec!["1-2a"]);

        assert_eq!(parse_to_vec("0-4/20a").unwrap(), vec!["0-4/20a"]);
        assert_eq!(
            parse_to_vec("0-4/2,0a").unwrap(),
            vec!["0", "0a", "2", "4",]
        );
    }

    #[test]
    fn test_nodeset_parse_with_suffix_component() {
        assert_eq!(parse_to_fold("a[1-3]_e,a4_e").unwrap(), "a[1-4]_e");
        assert_eq!(
            parse_to_vec("a[1-3]_e,a4_e").unwrap(),
            vec!["a1_e", "a2_e", "a3_e", "a4_e"]
        );

        assert_eq!(
            parse_to_fold("a[10-11]b[2-3]_cd").unwrap(),
            "a[10-11]b[2-3]_cd"
        );
        assert_eq!(
            parse_to_vec("a[10-11]b[2-3]_cd").unwrap(),
            vec!["a10b2_cd", "a10b3_cd", "a11b2_cd", "a11b3_cd"]
        );
    }

    #[test]
    fn test_nodeset_parse_with_affix_id() {
        assert_eq!(
            parse_to_vec("a10[1-3]").unwrap(),
            vec!["a101", "a102", "a103"]
        );

        assert_eq!(
            parse_to_vec("a[1-3]10").unwrap(),
            vec!["a110", "a210", "a310"]
        );

        assert_eq!(
            parse_to_vec("a10[1-3]10").unwrap(),
            vec!["a10110", "a10210", "a10310"]
        );

        assert_eq!(
            parse_to_vec("a00[1-3]00").unwrap(),
            vec!["a00100", "a00200", "a00300"]
        );

        assert_eq!(
            parse_to_vec("a20[0-100/20]5").unwrap(),
            vec!["a2005", "a201005", "a20205", "a20405", "a20605", "a20805"]
        );

        assert_eq!(
            parse_to_vec("a02[0-100/20]05").unwrap(),
            vec!["a02005", "a0210005", "a022005", "a024005", "a026005", "a028005",]
        );
    }

    #[test]
    fn test_nodeset_intersect() {
        let id1: NodeSet<IdRangeList> = "x[1-10/2,5]y[1-7]z3,x[1-10/2,5]y[1-7]z2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "x[2-5]y7z[2,3]".parse().unwrap();

        assert_eq!(id1.intersection(&id2).fold().to_string(), "x[3,5]y7z[2-3]");

        let id1: NodeSet<IdRangeList> = "a1 a2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "b1 b2".parse().unwrap();

        assert!(id1.intersection(&id2).is_empty(),);
    }

    #[test]
    fn test_nodeset_overflow() {
        assert!(parse_to_vec("a10000000000").is_err());
        assert!(parse_to_vec("a[9999999999-10000000000]").is_err());
        assert_eq!(parse_to_vec("a3183856184").unwrap(), vec!["a3183856184"]);
        assert!(parse_to_vec("a03183856185").is_err());
        assert!(parse_to_vec("a[3183856184-3183856185]").is_err());
        assert!(parse_to_vec("a00000000000").is_err());
        assert_eq!(parse_to_vec("a0000000000").unwrap(), vec!["a0000000000"]);
        assert_eq!(
            parse_to_vec("a[3183856183-3183856184]").unwrap(),
            vec!["a3183856183", "a3183856184"]
        );
        assert_eq!(
            parse_to_vec("a[3183856170-3183856184/10]").unwrap(),
            vec!["a3183856170", "a3183856180"]
        );

        assert_eq!(
            parse_to_vec("a[3183856170-3183856184/2000000000]").unwrap(),
            vec!["a3183856170"]
        );
        assert!(parse_to_vec("a[3183856170-3183856184/20000000000]").is_err(),);
    }

    #[test]
    fn test_nodeset_overflow_affix() {
        assert!(parse_to_vec("a10000[0,1]00000").is_err());
        assert!(parse_to_vec("a00000[0,1]00000").is_err());
        assert_eq!(
            parse_to_vec("n000000000[0-0]").unwrap(),
            vec!["n0000000000"]
        );
        assert!(parse_to_vec("n000000000[0-10]").is_err());
        assert_eq!(
            parse_to_fold("n00000000[0-10]").unwrap(),
            "n[000000000-000000009,0000000010]"
        );

        assert_eq!(
            parse_to_fold("n31838561[0-84]").unwrap(),
            "n[318385610-318385619,3183856110-3183856184]"
        );

        assert_eq!("a318385[0-61]84".parse::<NodeSet>().unwrap().len(), 62);
        assert!(parse_to_fold("a318385[0-62]84").is_err());
        assert!(parse_to_fold("a318385[0-61]85").is_err());

        assert_eq!("a[3183855-3183856]84".parse::<NodeSet>().unwrap().len(), 2);
        assert_eq!(
            parse_to_vec("a[31838560-31838561]84").unwrap(),
            vec!["a3183856084", "a3183856184"]
        );
    }

    #[test]
    fn test_nodeset_symmetric_diff() {
        let id1: NodeSet<IdRangeList> = "x[1-10/2,5]y[1-7]z3,x[1-10/2,5]y[1-7]z2".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "x[2-5]y7z[2,3]".parse().unwrap();

        assert_eq!(
            id1.symmetric_difference(&id2).fold().to_string(),
            "x[1,7,9]y[1-7]z[2-3],x[2,4]y7z[2-3],x[3,5]y[1-6]z[2-3]"
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
        let id5: NodeSet<IdRangeList> = "a[00-20]".parse().unwrap();

        assert_eq!(id1.len(), 2);
        assert_eq!(id2.len(), 10);
        assert_eq!(id3.len(), 90);
        assert_eq!(id4.len(), 10102);
        assert_eq!(id5.len(), 21);
    }

    #[test]
    fn test_nodeset_fold() {
        assert_eq!(
            parse_to_fold("a[1-10/2,5]b[1-7]c3,a[1-10/2,5]b[1-7]c2").unwrap(),
            "a[1,3,5,7,9]b[1-7]c[2-3]"
        );
        assert_eq!(
            parse_to_fold("a[0-10]b[0-10],a[0-20]b[0-10]").unwrap(),
            "a[0-20]b[0-10]"
        );
        assert_eq!(
            parse_to_fold("x[0-10]y[0-10],x[8-18]y[8-18],x[11-18]y[0-7]").unwrap(),
            "x[0-7]y[0-10],x[8-18]y[0-18]"
        );
    }

    #[test]
    fn test_nodeset_iter() {
        assert_eq!(
            parse_to_vec("a[1-2]b[1-2]").unwrap(),
            vec!["a1b1", "a1b2", "a2b1", "a2b2"]
        );
    }

    #[test]
    fn test_nodeset_padded() {
        assert_eq!(
            parse_to_fold("a01 a02 a9 a0 a00 a1 a09 a10").unwrap(),
            "a[0-1,9,00-02,09-10]"
        );

        assert_eq!(
            parse_to_fold("n[001-999] n[1000]").unwrap(),
            "n[001-999,1000]"
        );

        assert_eq!(
            parse_to_fold("n[000-999] n[1000] n1001 n0 n1").unwrap(),
            "n[0-1,000-999,1000-1001]"
        );

        assert_eq!(
            parse_to_fold("n[0000-1000] n[1000-10000] n1001 n1 n00").unwrap(),
            "n[1,00,0000-9999,10000]"
        );

        assert_eq!(
            parse_to_fold("n[0000000000-0000000001]").unwrap(),
            "n[0000000000-0000000001]"
        );
    }

    #[test]
    fn test_nodeset_parse_operators() {
        assert_eq!(
            parse_to_fold("node[0-10] - (node[0-5] + node[7-8], node9 node10)").unwrap(),
            "node6"
        );
        assert_eq!(parse_to_fold("node[1-2] ^ node[2-3]").unwrap(), "node[1,3]");
        assert_eq!(parse_to_fold("node[1-2]^node[2-3]").unwrap(), "node[1,3]");

        assert_eq!(parse_to_fold("node[1-2] ! node[2-3]").unwrap(), "node1");
        assert_eq!(parse_to_fold("node[1-2]!node[2-3]").unwrap(), "node1");
        assert_eq!(parse_to_fold("node[1-2] - node[2-3]").unwrap(), "node1");

        assert_eq!(
            parse_to_vec("node[1-2]-node[2-3]").unwrap(),
            vec!["node1-node2", "node1-node3", "node2-node2", "node2-node3"]
        );
    }

    #[test]
    fn test_rangeset_parse_operators() {
        assert_eq!(parse_to_fold("[0-10] - ([0-5] + [7-8],9 10)").unwrap(), "6");
        assert_eq!(parse_to_fold("0-10 - (0-5 + 7-8,9 10)").unwrap(), "6");
        assert_eq!(parse_to_fold("1-2 ^ 2-3").unwrap(), "1,3");
        assert_eq!(parse_to_fold("[1-2] ^ [2-3]").unwrap(), "1,3");
        assert_eq!(parse_to_fold("[1-2]^[2-3]").unwrap(), "1,3");
        assert_eq!(parse_to_fold("1-2^2-3").unwrap(), "1,3");

        assert_eq!(parse_to_fold("1-2 ! 2-3").unwrap(), "1");
        assert_eq!(parse_to_fold("[1-2] ! [2-3]").unwrap(), "1");
        assert_eq!(parse_to_fold("1-2 ! 2-3").unwrap(), "1");
        assert_eq!(parse_to_fold("[1-2] ! [2-3]").unwrap(), "1");
        assert_eq!(parse_to_fold("1-2!2-3").unwrap(), "1");
        assert_eq!(parse_to_fold("[1-2]![2-3]").unwrap(), "1");

        assert_eq!(parse_to_fold("1-2 - 2-3").unwrap(), "1");
        assert_eq!(parse_to_fold("[1-2] - [2-3]").unwrap(), "1");
        assert_eq!(
            parse_to_vec("[1-2]-[2-3]").unwrap(),
            vec!["1-2", "1-3", "2-2", "2-3"]
        );
    }

    #[test]
    fn test_nodeset_late_fold() {
        let mut id1: NodeSet<IdRangeList> = "a[1-2]b[1-2]".parse().unwrap();
        let id2: NodeSet<IdRangeList> = "a[1-2]b[1-4]".parse().unwrap();
        let id3: NodeSet<IdRangeList> = "a1b1".parse().unwrap();
        id1.extend_from_nodeset(&id2);

        let mut id4 = id1.difference(&id3);

        id4.extend_from_nodeset(&id3);
        id4.fold();
        assert_eq!(id4.to_string(), "a[1-2]b[1-4]",);
    }

    #[test]
    fn test_nodeset_multiple_bases() {
        assert_eq!(
            parse_to_vec("a1,n[40-41],y[100-101],[2-3]y").unwrap(),
            vec!["2y", "3y", "a1", "n40", "n41", "y100", "y101",]
        );
        assert_eq!(parse_to_vec("a1,a, 1a").unwrap(), vec!["1a", "a", "a1"]);
        assert_eq!(parse_to_vec("a1a,a1a1").unwrap(), vec!["a1a", "a1a1"]);
    }

    #[test]
    fn test_nodeset_sorted() {
        assert_eq!(
            "h1,z,a2,a0a,a1a1"
                .parse::<NodeSet>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec!["a2", "a1a1", "a0a", "h1", "z"]
        );
    }
}
