use crate::collections::idset::IdRangeProduct;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{char, digit1, multispace0, one_of},
    combinator::{all_consuming, map, map_res, opt, verify},
    multi::{fold_many0, many0, separated_list1},
    sequence::{delimited, pair, tuple},
    IResult,
};
use std::fmt;

fn is_component_char(c: char) -> bool {
    char::is_alphabetic(c) || ['-', '_', '.'].contains(&c)
}

use super::nodeset::NodeSetDimensions;
use crate::idrange::{IdRange, IdRangeStep};
use crate::{IdSet, NodeSet, NodeSetParseError};

fn term<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    delimited(
        multispace0,
        alt((group_or_nodeset, delimited(char('('), expr, char(')')))),
        multispace0,
    )(i)
}

pub fn op(i: &str) -> IResult<&str, char, CustomError<&str>> {
    delimited(multispace0, one_of("+,&!^"), multispace0)(i)
}

/* fn emptyset<T>(i: &str) -> IResult<&str, NodeSet<T>, VerboseError<&str>> {
    map(multispace0, |_| NodeSet::default())(i)
} */

pub fn full_expr<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    all_consuming(expr)(i)
}

pub fn expr<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    fold_many0(
        tuple((opt(op), term)),
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

fn group_or_nodeset<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    alt((group, nodeset))(i)
}

fn group<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    map_res(
        pair(char('@'), group_name),
        |r| -> Result<NodeSet<T>, NodeSetParseError> { r.1.parse() },
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
                    node_component,
                    alt((id_standalone, id_range_bracketed)),
                )),
                opt(node_component),
            ),
            |r| !r.0.is_empty() || r.1.is_some(),
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
                dims.push(dim, false);
            }
            if let Some(dim) = r.1 {
                dims.push(dim, true);
            }

            let mut ns = NodeSet::new();
            if ranges.is_empty() {
                ns.dimnames.entry(dims).or_insert_with(|| None);
            } else {
                ns.dimnames
                    .entry(dims)
                    .or_insert_with(|| Some(IdSet::new()))
                    .as_mut()
                    .unwrap()
                    .products
                    .push(IdRangeProduct { ranges });
            }
            ns.fold();
            ns
        },
    )(i)
}

fn group_name(i: &str) -> IResult<&str, &str, CustomError<&str>> {
    take_while1(is_component_char)(i)
}

fn node_component(i: &str) -> IResult<&str, &str, CustomError<&str>> {
    take_while1(is_component_char)(i)
}

fn id_range_bracketed(i: &str) -> IResult<&str, Vec<IdRangeStep>, CustomError<&str>> {
    delimited(
        char('['),
        separated_list1(tag(","), id_range_step),
        char(']'),
    )(i)
}

use std::convert::TryFrom;

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

use nom::combinator::cut;
use nom::error::ErrorKind;
use nom::error::FromExternalError;
use nom::error::ParseError;
use std::convert::TryInto;

#[derive(Debug, PartialEq, Eq)]
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

fn is_padded(s: &str) -> bool {
    s.chars().take_while(|c| *c == '0').count() > 0 && s != "0"
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
                            let mut padded = is_padded(s.0);

                            let (end, step) = match s.1 {
                                None => {(start, 1)},
                                Some(s1) => {

                                    padded |= is_padded(s1.1);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_range_step() {
        assert_eq!(
            id_range_step("2"),
            Ok((
                "",
                IdRangeStep {
                    start: 2,
                    end: 2,
                    step: 1,
                    pad: 0
                }
            ))
        );
        assert_eq!(
            id_range_step("2-34"),
            Ok((
                "",
                IdRangeStep {
                    start: 2,
                    end: 34,
                    step: 1,
                    pad: 0
                }
            ))
        );
        assert_eq!(
            id_range_step("2-34/8"),
            Ok((
                "",
                IdRangeStep {
                    start: 2,
                    end: 34,
                    step: 8,
                    pad: 0
                }
            ))
        );

        assert!(id_range_step("-34/8").is_err());
        assert!(id_range_step("/8").is_err());
        assert_eq!(
            id_range_step("34/8"),
            Ok((
                "/8",
                IdRangeStep {
                    start: 34,
                    end: 34,
                    step: 1,
                    pad: 0
                }
            ))
        );
    }

    #[test]
    fn test_id_range_bracketed() {
        assert_eq!(
            id_range_bracketed("[2]"),
            Ok((
                "",
                vec![IdRangeStep {
                    start: 2,
                    end: 2,
                    step: 1,
                    pad: 0
                }]
            ))
        );
        assert_eq!(
            id_range_bracketed("[2,3-4,5-67/8]"),
            Ok((
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
            ))
        );

        assert!(id_range_bracketed("[2,]").is_err());
        assert!(id_range_bracketed("[/8]").is_err());
        assert!(id_range_bracketed("[34-]").is_err());
    }

    #[test]
    fn test_node_component() {
        assert_eq!(node_component("abcd efg"), Ok((" efg", "abcd")));
        assert!(node_component(" abcdefg").is_err());
        assert_eq!(node_component("a_b-c.d2efg"), Ok(("2efg", "a_b-c.d")));
    }
}
