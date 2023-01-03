use super::parsers;
use crate::idrange::IdRange;
use crate::{IdSet, IdSetIter};
use itertools::Itertools;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NodeSet<T> {
    pub(crate) dimnames: HashMap<NodeSetDimensions, Option<IdSet<T>>>,
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
    dim_iter: std::iter::Peekable<
        std::collections::hash_map::Iter<'a, NodeSetDimensions, Option<IdSet<T>>>,
    >,
    set_iter: Option<IdSetIter<'a, T>>,
}

impl<'b, T> NodeSetIter<'b, T>
where
    T: IdRange + fmt::Display + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    fn new(dims: &'b HashMap<NodeSetDimensions, Option<IdSet<T>>>) -> Self {
        let mut it = Self {
            dim_iter: dims.iter().peekable(),
            set_iter: None,
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
            .and_then(|dims| dims.1.as_ref())
            .map(|s| s.iter());
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
            let Some(set_iter) = self.set_iter.as_mut() else {
                self.next_dims();
                return Some(dimnames.iter().join(""));
            };

            if let Some(coords) = set_iter.next() {
                /* println!("next coord"); */
                return Some(
                    dimnames
                        .iter()
                        .zip(coords.iter())
                        .map(|(a, b)| format!("{}{}", a, b))
                        .join(""),
                );
            } else {
                /* println!("next dim"); */
                self.next_dims();
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
            .iter()
            .map(|(_, set)| set.as_ref().map(|s| s.len()).unwrap_or(1))
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.dimnames.is_empty()
    }

    pub fn iter(&self) -> NodeSetIter<'_, T> {
        /* println!("{:?}", self.dimnames); */
        NodeSetIter::new(&self.dimnames)
    }

    pub fn fold(&mut self) -> &mut Self {
        /* println!("fold {:?}", self.dimnames); */
        self.dimnames.values_mut().for_each(|s| {
            if let Some(s) = s {
                s.fold();
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
                Some(set) => {
                    if let Some(s) = set {
                        s.extend(oset.as_ref().unwrap())
                    }
                }
            };
        }
    }

    pub fn difference(&self, other: &Self) -> Self {
        let mut dimnames = HashMap::<NodeSetDimensions, Option<IdSet<T>>>::new();
        /*    println!("Start intersect"); */
        for (dimname, set) in self.dimnames.iter() {
            /*    println!("{:?}", dimname); */
            if let Some(oset) = other.dimnames.get(dimname) {
                /*    println!("Same dims"); */
                match (set, oset) {
                    (None, None) => continue,
                    (Some(set), Some(oset)) => {
                        if let Some(nset) = set.difference(oset) {
                            /*  println!("Intersect"); */
                            dimnames.insert(dimname.clone(), Some(nset));
                        }
                    }
                    _ => {
                        dimnames.insert(dimname.clone(), set.clone());
                    }
                }
            } else {
                dimnames.insert(dimname.clone(), set.clone());
            }
        }
        NodeSet { dimnames }
    }

    pub fn intersection(&self, other: &Self) -> Self {
        let mut dimnames = HashMap::<NodeSetDimensions, Option<IdSet<T>>>::new();
        /*    println!("Start intersect"); */
        for (dimname, set) in self.dimnames.iter() {
            /*    println!("{:?}", dimname); */
            if let Some(oset) = other.dimnames.get(dimname) {
                /*    println!("Same dims"); */
                match (set, oset) {
                    (None, None) => {
                        dimnames.insert(dimname.clone(), None);
                    }
                    (Some(set), Some(oset)) => {
                        if let Some(nset) = set.intersection(oset) {
                            /*  println!("Intersect"); */
                            dimnames.insert(dimname.clone(), Some(nset));
                        }
                    }
                    _ => continue,
                }
            }
        }

        NodeSet { dimnames }
    }

    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut dimnames = HashMap::<NodeSetDimensions, Option<IdSet<T>>>::new();
        /*    println!("Start intersect"); */
        for (dimname, set) in self.dimnames.iter() {
            /*    println!("{:?}", dimname); */
            if let Some(oset) = other.dimnames.get(dimname) {
                /*    println!("Same dims"); */
                match (set, oset) {
                    (None, None) => continue,
                    (Some(set), Some(oset)) => {
                        if let Some(nset) = set.symmetric_difference(oset) {
                            /*  println!("Intersect"); */
                            dimnames.insert(dimname.clone(), Some(nset));
                        }
                    }
                    (Some(set), None) => {
                        dimnames.insert(dimname.clone(), Some(set.clone()));
                    }
                    (None, Some(oset)) => {
                        dimnames.insert(dimname.clone(), Some(oset.clone()));
                    }
                }
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
        parsers::full_expr::<T>(s)
            .map(|r| r.1)
            .map_err(|e| match e {
                nom::Err::Error(e) => NodeSetParseError::new(nom::error::convert_error(s, e)),
                _ => panic!("unreachable"),
            })
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
            if let Some(set) = set {
                set.fmt_dims(f, &dim.dimnames)
                    .expect("failed to format string");
            } else {
                write!(f, "{}", dim.dimnames[0])?;
            }
            first = false;
        }
        Ok(())
    }
}

pub struct NodeSetParseError {
    err: String,
}

impl NodeSetParseError {
    fn new(err: String) -> NodeSetParseError {
        NodeSetParseError { err }
    }
}

impl fmt::Debug for NodeSetParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.err)
    }
}

impl fmt::Display for NodeSetParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.err)
    }
}

use nom::error::VerboseError;

impl From<nom::Err<VerboseError<&str>>> for NodeSetParseError {
    fn from(error: nom::Err<VerboseError<&str>>) -> Self {
        NodeSetParseError::new(format!("{:?}", error))
    }
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
