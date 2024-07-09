use super::config::Resolver;
use crate::collections::idset::IdRangeProduct;
use crate::collections::nodeset::IdSetKind;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{char, digit1, multispace0, multispace1, one_of},
    combinator::{all_consuming, map, map_res, opt, value, verify},
    multi::{fold_many0, many0, separated_list1},
    sequence::{delimited, pair, separated_pair, tuple},
    IResult,
};
use std::fmt;

fn is_component_char(c: char) -> bool {
    char::is_alphabetic(c) || ['-', '_', '.'].contains(&c)
}

use super::nodeset::NodeSetDimensions;
use crate::idrange::{IdRange, IdRangeStep};
use crate::{IdSet, NodeSet, NodeSetParseError};

/// Parse strings into nodesets
#[derive(Copy, Clone, Default)]
pub struct Parser<'a> {
    resolver: Option<&'a Resolver>,
    default_source: Option<&'a str>,
}

impl<'a> Parser<'a> {
    /// Create a new parser from a resolver
    pub fn with_resolver(resolver: &'a Resolver, default_source: Option<&'a str>) -> Self {
        Self {
            resolver: Some(resolver),
            default_source,
        }
    }

    /// Parse a string into a nodeset
    pub fn parse<T>(self, i: &str) -> Result<NodeSet<T>, NodeSetParseError>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        let mut ns = all_consuming(|i| self.expr(i))(i)
            .map(|r| r.1)
            .map_err(|e| match e {
                nom::Err::Error(e) => NodeSetParseError::from(e),
                nom::Err::Failure(e) => NodeSetParseError::from(e),
                _ => panic!("unreachable"),
            })?;

        ns.fold();
        Ok(ns)
    }

    fn expr<T>(self, i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        delimited(
            multispace0,
            map_res(
                fold_many0(
                    tuple((|i| self.term(i), opt(Self::op))),
                    || (None, None, false),
                    |acc, t| {
                        let (ns, op, _) = acc;

                        let Some(mut ns) = ns else {
                            return (Some(t.0), t.1, false);
                        };

                        let Some(op) = op else {
                            return (Some(ns), t.1, true);
                        };

                        match op {
                            ',' | '+' | ' ' => {
                                ns.extend_from_nodeset(&t.0);
                            }
                            '!' | '-' => {
                                ns = ns.difference(&t.0);
                            }
                            '^' => {
                                ns = ns.symmetric_difference(&t.0);
                            }
                            '&' => {
                                ns = ns.intersection(&t.0);
                            }
                            _ => unreachable!(),
                        }
                        (Some(ns), t.1, false)
                    },
                ),
                |(ns, _, err)| {
                    if err {
                        Err(NodeSetParseError::Generic(i.to_string()))
                    } else {
                        Ok(ns.unwrap_or_default())
                    }
                },
            ),
            multispace0,
        )(i)
    }

    fn term<T>(self, i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        alt((
            |i| self.group_or_nodeset(i),
            delimited(char('('), |i| self.expr(i), char(')')),
        ))(i)
    }

    fn op(i: &str) -> IResult<&str, char, CustomError<&str>> {
        alt((
            delimited(multispace0, one_of("+,&!^"), multispace0),
            delimited(multispace1, one_of("-"), multispace1),
            value(' ', multispace1),
        ))(i)
    }

    fn group_or_nodeset<T>(self, i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        alt((Self::nodeset, |i| self.group(i), Self::rangeset))(i)
    }

    fn rangeset<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        map(
            separated_list1(tag(","), Self::id_range_step),
            |idrs_list: Vec<IdRangeStep>| {
                let mut ns = NodeSet::lazy();
                let mut dims = NodeSetDimensions::new();
                dims.push("");
                let mut id = T::new().lazy();
                for idrs in idrs_list {
                    id.push_idrs(&idrs);
                }
                id.sort();
                ns.dimnames.entry(dims).or_insert(IdSetKind::Single(id));
                ns
            },
        )(i)
    }

    fn nodeset<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        map(
            verify(
                pair(
                    many0(pair(
                        Self::node_component,
                        alt((Self::id_standalone, Self::id_range_bracketed)),
                    )),
                    opt(Self::node_component),
                ),
                |r| {
                    let first_component = r.0.first().map(|first| first.0).or(r.1);
                    first_component
                        .and_then(|s| s.chars().next())
                        .map(|c| c.is_alphabetic())
                        .unwrap_or(false)
                },
            ),
            |r| {
                let mut dims = NodeSetDimensions::new();
                let mut ranges = vec![];
                for (dim, rng) in r.0.into_iter() {
                    let mut range = T::new().lazy();
                    for r in rng {
                        range.push_idrs(&r)
                    }
                    range.sort();
                    ranges.push(range);
                    dims.push(dim);
                }
                if let Some(dim) = r.1 {
                    dims.push(dim);
                }

                let mut ns = NodeSet::lazy();
                if ranges.is_empty() {
                    ns.dimnames.entry(dims).or_insert_with(|| IdSetKind::None);
                } else if ranges.len() == 1 {
                    ns.dimnames
                        .entry(dims)
                        .or_insert(IdSetKind::Single(ranges.pop().unwrap()));
                } else {
                    let mut ids = IdSet::new();
                    ids.products.push(IdRangeProduct { ranges });
                    ns.dimnames.entry(dims).or_insert(IdSetKind::Multiple(ids));
                }

                ns
            },
        )(i)
    }

    fn group<T>(self, i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        map(
            pair(
                char('@'),
                cut(map_res(
                    // Match either 'sources:groups' or 'groups'
                    // Both sources and groups can be sets i.e: @source[1-4]:group[1,5]
                    alt((
                        map(tag("*"), |_| (None, None)),
                        |s| self.group_with_source(s),
                        map(Self::nodeset, |s: NodeSet<T>| (None, Some(s))),
                    )),
                    |(sources, groups)| -> Result<NodeSet<T>, NodeSetParseError> {
                        let mut ns = NodeSet::lazy();

                        let Some(resolver) = self.resolver else {
                            return Ok(ns);
                        };

                        // Iterate over the sources or use an iterator
                        // which yields the default source once
                        for source in sources
                            .as_ref()
                            .map(|sources| -> Box<dyn Iterator<Item = Option<String>>> {
                                Box::new(sources.iter().map(Some))
                            })
                            .unwrap_or_else(|| -> Box<dyn Iterator<Item = Option<String>>> {
                                Box::new(std::iter::once(
                                    self.default_source.map(|s| s.to_string()),
                                ))
                            })
                        {
                            let all_groups;
                            let groups = match &groups {
                                Some(groups) => groups,
                                None => {
                                    all_groups = resolver.list_groups(source.as_deref());
                                    &all_groups
                                }
                            };

                            for group in groups.iter() {
                                let nodeset = resolver.resolve(source.as_deref(), &group)?;
                                ns.extend_from_nodeset(&nodeset);
                            }
                        }

                        Ok(ns)
                    },
                )),
            ),
            |r| r.1,
        )(i)
    }

    #[allow(clippy::type_complexity)]
    fn group_with_source<T>(
        self,
        i: &str,
    ) -> IResult<&str, (Option<NodeSet<T>>, Option<NodeSet<T>>), CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        alt((
            map(pair(Self::nodeset, tag(":*")), |source| {
                (Some(source.0), None)
            }),
            map(
                separated_pair(Self::nodeset, char(':'), opt(Self::nodeset)),
                |source| (Some(source.0), Some(source.1.unwrap_or_default())),
            ),
        ))(i)
    }

    fn node_component(i: &str) -> IResult<&str, &str, CustomError<&str>> {
        take_while1(is_component_char)(i)
    }

    fn id_range_bracketed(i: &str) -> IResult<&str, Vec<IdRangeStep>, CustomError<&str>> {
        delimited(
            char('['),
            cut(separated_list1(tag(","), Self::id_range_step)),
            char(']'),
        )(i)
    }

    fn id_standalone(i: &str) -> IResult<&str, Vec<IdRangeStep>, CustomError<&str>> {
        map_res(
            digit1,
            |d: &str| -> Result<Vec<IdRangeStep>, NodeSetParseError> {
                let start = d.parse::<u32>()?;
                Ok(vec![IdRangeStep {
                    start,
                    end: start,
                    step: 1,
                    pad: u32::try_from(d.len())?,
                }])
            },
        )(i)
    }

    #[allow(clippy::type_complexity)]
    fn id_range_step(i: &str) -> IResult<&str, IdRangeStep, CustomError<&str>> {
        map_res(pair(digit1,
                         opt(tuple((
                                 tag("-"),
                                digit1,
                                opt(
                                    pair(
                                        tag("/"),
                                        digit1)))))),

                        |s: (&str, Option<(&str, &str, Option<(&str, &str)>)>) | -> Result<IdRangeStep, NodeSetParseError>{
                            let start = s.0.parse::<u32>()?;
                            let mut padded = Self::is_padded(s.0);

                            let (end, step) = match s.1 {
                                None => {(start, 1)},
                                Some(s1) => {

                                    padded |= Self::is_padded(s1.1);
                                    if padded && s1.1.len() != s.0.len() {
                                        return Err(NodeSetParseError::Padding(i.to_string()));
                                    }

                                    let end = s1.1.parse::<u32>() ?;
                                    match s1.2 {
                                        None => {(end, 1)},
                                        Some((_, step)) => {(end, step.parse::<usize>()?)}
                                    }
                                }
                            };
                            let pad = if padded {
                                s.0.len()
                            } else {
                                0
                            };
                            if start > end {
                                return Err(NodeSetParseError::Reverse(i.to_string()));
                            }

                            Ok(IdRangeStep{start, end, step, pad: pad.try_into()?})
                        }
                        )(i)
    }

    fn is_padded(s: &str) -> bool {
        s.starts_with('0') && s != "0"
    }
}

use std::convert::TryFrom;

use nom::combinator::cut;
use nom::error::ErrorKind;
use nom::error::FromExternalError;
use nom::error::ParseError;
use std::convert::TryInto;

#[derive(Debug)]
pub enum CustomError<I> {
    NodeSetError(NodeSetParseError),
    Nom(I, ErrorKind),
}

impl<I> ParseError<I> for CustomError<I> {
    fn from_error_kind(input: I, kind: ErrorKind) -> Self {
        CustomError::Nom(input, kind)
    }

    fn append(_: I, _: ErrorKind, other: Self) -> Self {
        other
    }
}
impl<I> FromExternalError<I, NodeSetParseError> for CustomError<I> {
    fn from_external_error(_: I, _: ErrorKind, e: NodeSetParseError) -> Self {
        CustomError::NodeSetError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nodeset_first_letter() {
        // a nodeset can only begin with '@' or an alphabetic character.
        assert!("-ab".parse::<NodeSet>().is_err());
        assert!("1ab".parse::<NodeSet>().is_err());
    }

    #[test]
    fn test_id_range_step() {
        assert_eq!(
            Parser::id_range_step("2").unwrap(),
            (
                "",
                IdRangeStep {
                    start: 2,
                    end: 2,
                    step: 1,
                    pad: 0
                }
            )
        );
        assert_eq!(
            Parser::id_range_step("2-34").unwrap(),
            (
                "",
                IdRangeStep {
                    start: 2,
                    end: 34,
                    step: 1,
                    pad: 0
                }
            )
        );
        assert_eq!(
            Parser::id_range_step("2-34/8").unwrap(),
            (
                "",
                IdRangeStep {
                    start: 2,
                    end: 34,
                    step: 8,
                    pad: 0
                }
            )
        );

        assert!(Parser::id_range_step("-34/8").is_err());
        assert!(Parser::id_range_step("/8").is_err());
        assert_eq!(
            Parser::id_range_step("34/8").unwrap(),
            (
                "/8",
                IdRangeStep {
                    start: 34,
                    end: 34,
                    step: 1,
                    pad: 0
                }
            )
        );
    }

    #[test]
    fn test_id_range_bracketed() {
        assert_eq!(
            Parser::id_range_bracketed("[2]").unwrap(),
            (
                "",
                vec![IdRangeStep {
                    start: 2,
                    end: 2,
                    step: 1,
                    pad: 0
                }]
            )
        );
        assert_eq!(
            Parser::id_range_bracketed("[2,3-4,5-67/8]").unwrap(),
            (
                "",
                vec![
                    IdRangeStep {
                        start: 2,
                        end: 2,
                        step: 1,
                        pad: 0
                    },
                    IdRangeStep {
                        start: 3,
                        end: 4,
                        step: 1,
                        pad: 0
                    },
                    IdRangeStep {
                        start: 5,
                        end: 67,
                        step: 8,
                        pad: 0
                    }
                ]
            )
        );

        assert!(Parser::id_range_bracketed("[2,]").is_err());
        assert!(Parser::id_range_bracketed("[/8]").is_err());
        assert!(Parser::id_range_bracketed("[34-]").is_err());
    }

    #[test]
    fn test_node_component() {
        assert_eq!(
            Parser::node_component("abcd efg").unwrap(),
            (" efg", "abcd")
        );
        assert!(Parser::node_component(" abcdefg").is_err());
        assert_eq!(
            Parser::node_component("a_b-c.d2efg").unwrap(),
            ("2efg", "a_b-c.d")
        );
    }
}
