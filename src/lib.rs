#![doc = include_str!("../README.md")]
mod collections;
mod idrange;

pub use idrange::IdRangeTree;
pub use idrange::IdRangeList;
pub use collections::IdSet;
pub use collections::NodeSet;
pub use collections::IdSetIter;
pub use collections::NodeSetParseError;
pub use collections::Resolver;
pub use collections::Parser;
