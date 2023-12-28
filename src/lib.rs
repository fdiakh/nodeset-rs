#![doc = include_str!("../README.md")]
mod collections;
mod idrange;

pub use collections::IdSet;
pub use collections::IdSetIter;
pub use collections::NodeSet;
pub use collections::NodeSetParseError;
pub use collections::Parser;
pub use collections::Resolver;
pub use idrange::IdRangeList;
pub use idrange::IdRangeTree;
