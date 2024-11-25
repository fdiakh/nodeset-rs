#![doc = include_str!("../README.md")]
mod collections;
mod idrange;

pub(crate) use collections::IdSet;
pub(crate) use collections::IdSetIter;
pub use collections::NodeSet;
pub use collections::NodeSetIter;
pub use collections::NodeSetParseError;
pub use collections::Parser;
pub use collections::Resolver;
pub use idrange::IdRangeList;
pub use idrange::IdRangeTree;
