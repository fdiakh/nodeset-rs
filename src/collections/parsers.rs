use super::config::Resolver;
use crate::collections::idset::IdRangeProduct;
use crate::collections::nodeset::IdSetKind;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{alpha1, char, digit1, multispace0, one_of},
    combinator::{all_consuming, map, map_res, opt, recognize, verify},
    multi::{fold_many0, many0, separated_list1},
    sequence::{delimited, pair, separated_pair, tuple},
    IResult,
};
use std::fmt;

fn is_component_char(c: char) -> bool {
    char::is_alphabetic(c) || ['-', '_', '.'].contains(&c)
}

fn is_component_alphanumeric(c: char) -> bool {
    char::is_alphanumeric(c) || ['-', '_', '.'].contains(&c)
}

use super::nodeset::NodeSetDimensions;
use crate::idrange::{IdRange, IdRangeStep};
use crate::{IdSet, NodeSet, NodeSetParseError};

#[derive(Copy, Clone, Default)]
pub struct Parser<'a> {
    resolver: Option<&'a Resolver>,
    default_source: Option<&'a str>,
}

impl<'a> Parser<'a> {
    pub fn with_resolver(resolver: &'a Resolver, default_source: Option<&'a str>) -> Self {
        Self {
            resolver: Some(resolver),
            default_source,
        }
    }

    pub fn parse<T>(self, i: &str) -> Result<NodeSet<T>, NodeSetParseError>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        all_consuming(|i| self.expr(i))(i)
            .map(|r| r.1)
            .map_err(|e| match e {
                nom::Err::Error(e) => NodeSetParseError::from(e),
                nom::Err::Failure(e) => NodeSetParseError::from(e),
                _ => panic!("unreachable"),
            })
    }

    fn expr<T>(self, i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        fold_many0(
            tuple((opt(Self::op), |i| self.term(i))),
            NodeSet::<T>::default,
            |mut ns, t| {
                match t.0 {
                    Some(',') | Some('+') | None => {
                        ns.extend(&t.1);
                    }
                    Some('!') => {
                        ns = ns.difference(&t.1);
                    }
                    Some('^') => {
                        ns = ns.symmetric_difference(&t.1);
                    }
                    Some('&') => {
                        ns = ns.intersection(&t.1);
                    }
                    _ => unreachable!(),
                }
                ns
            },
        )(i)
    }

    fn term<T>(self, i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        delimited(
            multispace0,
            alt((
                |i| self.group_or_nodeset(i),
                delimited(char('('), |i| self.expr(i), char(')')),
            )),
            multispace0,
        )(i)
    }

    fn op(i: &str) -> IResult<&str, char, CustomError<&str>> {
        delimited(multispace0, one_of("+,&!^"), multispace0)(i)
    }

    fn group_or_nodeset<T>(self, i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        alt((Self::nodeset, |i| self.group(i)))(i)
    }

    fn nodeset<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        map_res(
            verify(
                pair(
                    many0(pair(
                        Self::node_component,
                        alt((Self::id_standalone, Self::id_range_bracketed)),
                    )),
                    opt(Self::node_component),
                ),
                |r| !r.0.is_empty() || r.1.is_some(),
            ),
            |r| -> Result<NodeSet<T>, NodeSetParseError> {
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

                let mut ns = NodeSet::default();
                if ranges.is_empty() {
                    ns.dimnames.entry(dims).or_insert_with(|| IdSetKind::None);
                } else if ranges.len() == 1 {
                    let IdSetKind::Single(id) = ns
                        .dimnames
                        .entry(dims)
                        .or_insert_with(|| IdSetKind::Single(T::new()))
                    else {
                        panic!("mismatched dimensions")
                    };
                    id.push(&ranges[0]);
                } else {
                    let IdSetKind::Multiple(id) = ns
                        .dimnames
                        .entry(dims)
                        .or_insert_with(|| IdSetKind::Multiple(IdSet::new()))
                    else {
                        panic!("mismatched dimensions")
                    };
                    id.products.push(IdRangeProduct { ranges });
                }
                ns.fold();
                Ok(ns)
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
                    alt((
                        Self::group_with_source,
                        map(Self::group_identifier, |s| (None, s)),
                    )),
                    |(source, group)| -> Result<NodeSet<T>, NodeSetParseError> {
                        if let Some(resolver) = self.resolver {
                            resolver.resolve(source.or(self.default_source), group)
                        } else {
                            Ok(NodeSet::default())
                        }
                    },
                )),
            ),
            |r| r.1,
        )(i)
    }

    fn group_with_source(i: &str) -> IResult<&str, (Option<&str>, &str), CustomError<&str>> {
        map(
            separated_pair(Self::group_identifier, char(':'), Self::group_identifier),
            |r| (Some(r.0), r.1),
        )(i)
    }

    fn group_identifier(i: &str) -> IResult<&str, &str, CustomError<&str>> {
        recognize(pair(alpha1, take_while(is_component_alphanumeric)))(i)
    }

    fn node_component(i: &str) -> IResult<&str, &str, CustomError<&str>> {
        take_while1(is_component_char)(i)
    }

    fn id_range_bracketed(i: &str) -> IResult<&str, Vec<IdRangeStep>, CustomError<&str>> {
        delimited(
            char('['),
            separated_list1(tag(","), Self::id_range_step),
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
        cut(map_res(pair(digit1,
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
                        ))(i)
    }

    fn is_padded(s: &str) -> bool {
        s.chars().take_while(|c| *c == '0').count() > 0 && s != "0"
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
    fn test_group_identifier() {
        assert_eq!(Parser::group_identifier("a").unwrap(), ("", "a"));
        assert_eq!(Parser::group_identifier("a_b-c").unwrap(), ("", "a_b-c"));
        assert_eq!(Parser::group_identifier("ab 2").unwrap(), (" 2", "ab"));
        assert!(Parser::group_identifier("-ab").is_err());
        assert!(Parser::group_identifier("1ab").is_err());
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
