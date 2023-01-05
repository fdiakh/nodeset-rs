use crate::collections::idset::IdRangeProduct;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{char, digit1, multispace0, one_of},
    combinator::{all_consuming, map, map_res, opt, verify},
    error::VerboseError,
    multi::{fold_many0, many0, separated_list1},
    sequence::{delimited, pair, tuple},
    IResult,
};
use std::fmt;
use std::num::ParseIntError;

fn is_component_char(c: char) -> bool {
    char::is_alphabetic(c) || ['-', '_', '.'].contains(&c)
}

use super::nodeset::NodeSetDimensions;
use crate::idrange::{IdRange, IdRangeStep};
use crate::{IdSet, NodeSet};

fn term<T>(i: &str) -> IResult<&str, NodeSet<T>, VerboseError<&str>>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    delimited(
        multispace0,
        alt((nodeset, delimited(char('('), expr, char(')')))),
        multispace0,
    )(i)
}

pub fn op(i: &str) -> IResult<&str, char, VerboseError<&str>> {
    delimited(multispace0, one_of("+,&!^"), multispace0)(i)
}

/* fn emptyset<T>(i: &str) -> IResult<&str, NodeSet<T>, VerboseError<&str>> {
    map(multispace0, |_| NodeSet::default())(i)
} */

pub fn full_expr<T>(i: &str) -> IResult<&str, NodeSet<T>, VerboseError<&str>>
where
    T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
{
    all_consuming(expr)(i)
}

pub fn expr<T>(i: &str) -> IResult<&str, NodeSet<T>, VerboseError<&str>>
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

fn nodeset<T>(i: &str) -> IResult<&str, NodeSet<T>, VerboseError<&str>>
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
                let mut range = T::new_empty();
                for r in rng {
                    range.push_idrs(&r)
                }
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

fn node_component(i: &str) -> IResult<&str, &str, VerboseError<&str>> {
    take_while1(is_component_char)(i)
}

fn id_range_bracketed(i: &str) -> IResult<&str, Vec<IdRangeStep>, VerboseError<&str>> {
    delimited(
        char('['),
        separated_list1(tag(","), id_range_step),
        char(']'),
    )(i)
}

fn id_standalone(i: &str) -> IResult<&str, Vec<IdRangeStep>, VerboseError<&str>> {
    map_res(
        digit1,
        |d: &str| -> Result<Vec<IdRangeStep>, ParseIntError> {
            let start = d.parse::<u32>()?;
            Ok(vec![IdRangeStep {
                start,
                end: start,
                step: 1,
            }])
        },
    )(i)
}

#[allow(clippy::type_complexity)]
fn id_range_step(i: &str) -> IResult<&str, IdRangeStep, VerboseError<&str>> {
    map_res(pair(digit1,
                         opt(tuple((
                                 tag("-"),
                                digit1,
                                opt(
                                    pair(
                                        tag("/"),
                                        digit1)))))),

                        |s: (&str, Option<(&str, &str, Option<(&str, &str)>)>) | -> Result<IdRangeStep, ParseIntError>{
                            let start = s.0.parse::<u32>() ?;
                            let (end, step) = match s.1 {
                                None => {(start, 1)},
                                Some(s1) => {
                                    let end = s1.1.parse::<u32>() ?;
                                    match s1.2 {
                                        None => {(end, 1)},
                                        Some((_, step)) => {(end, step.parse::<usize>()?)}
                                    }
                                }
                            };
                            Ok(IdRangeStep{start, end, step})
                        }
                        )(i)
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
                    step: 1
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
                    step: 1
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
                    step: 8
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
                    step: 1
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
                    step: 1
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
                        step: 1
                    },
                    IdRangeStep {
                        start: 3,
                        end: 4,
                        step: 1
                    },
                    IdRangeStep {
                        start: 5,
                        end: 67,
                        step: 8
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
