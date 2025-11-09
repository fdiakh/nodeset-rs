use super::config::Resolver;
use super::nodeset::NodeSetDimensions;
use crate::collections::idset::IdRangeProduct;
use crate::collections::nodeset::IdSetKind;
use crate::idrange::{AffixIdRangeStep, IdRange, IdRangeOffset, IdRangeStep, SingleId};
use crate::{IdSet, NodeSet, NodeSetParseError};
use auto_enums::auto_enum;
use nom::bytes::complete::is_a;
use nom::combinator::{cut, eof, peek};
use nom::error::ErrorKind;
use nom::error::FromExternalError;
use nom::error::ParseError;
use nom::Parser as NomParser;

use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{char, digit1, multispace0, multispace1, one_of},
    combinator::{all_consuming, map, map_res, opt, value, verify},
    multi::{fold_many0, many0, separated_list1},
    sequence::{delimited, pair, separated_pair},
    IResult,
};
use std::convert::TryInto;

use std::fmt;

fn is_nodeset_char(c: char) -> bool {
    is_source_char(c) || c == ':'
}

fn is_source_char(c: char) -> bool {
    char::is_alphabetic(c) || ['-', '_', '.', '/', '+'].contains(&c)
}

/// Parse strings into nodesets
#[derive(Debug, Copy, Clone, Default)]
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
        let mut ns = all_consuming(|i| self.expr(i))
            .parse(i)
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
                    (|i| self.term(i), opt(Self::op)),
                    || (None, None, false),
                    |acc, t| {
                        let (ns, op, err) = acc;

                        if err {
                            return (None, None, true);
                        }

                        let Some(mut ns) = ns else {
                            return (Some(t.0), t.1, false);
                        };

                        let Some(op) = op else {
                            return (None, None, true);
                        };

                        match op {
                            ',' | ' ' => {
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
        )
        .parse(i)
    }

    fn term<T>(self, i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        alt((
            |i| self.group_or_nodeset(i),
            delimited(char('('), |i| self.expr(i), char(')')),
        ))
        .parse(i)
    }

    fn op(i: &str) -> IResult<&str, char, CustomError<&str>> {
        alt((
            delimited(multispace0, one_of(",&!^"), multispace0),
            delimited(multispace1, one_of("-"), multispace1),
            value(' ', multispace1),
        ))
        .parse(i)
    }

    fn group_or_nodeset<T>(self, i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        alt((Self::rangeset, Self::nodeset, |i| self.group(i))).parse(i)
    }

    fn nodeset_or_rangeset<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        alt((Self::nodeset, Self::rangeset)).parse(i)
    }

    fn rangeset<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        map_res(
            pair(
                alt((Self::id_range_bracketed_affix, Self::id_range_step_rangeset)),
                peek(alt((is_a(",&!^()"), multispace1, eof))),
            ),
            |(idrs, _)| {
                let mut ns = NodeSet::lazy();
                let mut dims = NodeSetDimensions::new();
                dims.push("");
                let mut range = T::new().lazy();

                match idrs {
                    IdRangeComponent::Single(id) => range.push_idrs(id),
                    IdRangeComponent::IdRange((high, rng, low)) => {
                        for r in rng {
                            range.push_idrs(AffixIdRangeStep::new(r, low, high)?)
                        }
                    }
                }

                range.sort();
                ns.bases.entry(dims).or_insert(IdSetKind::Single(range));
                Ok(ns)
            },
        )
        .parse(i)
    }

    fn nodeset<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        Self::set(i, false)
    }

    fn sourceset<T>(i: &str) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        Self::set(i, true)
    }

    fn set<T>(i: &str, source: bool) -> IResult<&str, NodeSet<T>, CustomError<&str>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        let parser = if source {
            Self::source_component
        } else {
            Self::node_component
        };

        map_res(
            verify(
                (
                    opt(alt((Self::id_range_bracketed_affix, Self::id_standalone))),
                    many0(pair(
                        parser,
                        alt((Self::id_range_bracketed_affix, Self::id_standalone)),
                    )),
                    opt(parser),
                ),
                |(_, components, suffix)| {
                    // This is a rangeset
                    if components.is_empty() && suffix.is_none() {
                        return false;
                    }

                    true
                },
            ),
            |(prefix, components, suffix)| {
                let mut dims = NodeSetDimensions::new();
                let mut ranges = vec![];

                let it = prefix
                    .into_iter()
                    .map(|prefix| ("", prefix))
                    .chain(components);

                for (dim, rng) in it {
                    let mut range = T::new().lazy();

                    match rng {
                        IdRangeComponent::Single(id) => range.push_idrs(id),
                        IdRangeComponent::IdRange((high, rng, low)) => {
                            for r in rng {
                                range.push_idrs(AffixIdRangeStep::new(r, low, high)?)
                            }
                        }
                    }
                    range.sort();
                    ranges.push(range);
                    dims.push(dim);
                }
                if let Some(dim) = suffix {
                    dims.push_suffix(dim);
                }

                let mut ns = NodeSet::lazy();
                if ranges.is_empty() {
                    ns.bases.entry(dims).or_insert_with(|| IdSetKind::None);
                } else if ranges.len() == 1 {
                    ns.bases
                        .entry(dims)
                        .or_insert(IdSetKind::Single(ranges.pop().unwrap()));
                } else {
                    let mut ids = IdSet::new();
                    ids.products.push(IdRangeProduct { ranges });
                    ns.bases.entry(dims).or_insert(IdSetKind::Multiple(ids));
                }

                Ok(ns)
            },
        )
        .parse(i)
    }

    #[auto_enum]
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
                        map(Self::nodeset_or_rangeset, |s: NodeSet<T>| (None, Some(s))),
                    )),
                    |(sources, groups)| -> Result<NodeSet<T>, NodeSetParseError> {
                        let mut ns = NodeSet::lazy();

                        let Some(resolver) = self.resolver else {
                            return Ok(ns);
                        };

                        #[auto_enum(Iterator)]
                        let sources = match &sources {
                            Some(sources) => sources.iter().map(Some),
                            None => std::iter::once(self.default_source.map(|s| s.to_string())),
                        };

                        // Iterate over the sources or use an iterator
                        // which yields the default source once
                        for source in sources {
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
        )
        .parse(i)
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
            map(pair(Self::sourceset, tag(":*")), |source| {
                (Some(source.0), None)
            }),
            map(
                separated_pair(Self::sourceset, char(':'), opt(Self::nodeset_or_rangeset)),
                |source| (Some(source.0), Some(source.1.unwrap_or_default())),
            ),
        ))
        .parse(i)
    }

    fn node_component(i: &str) -> IResult<&str, &str, CustomError<&str>> {
        take_while1(is_nodeset_char)(i)
    }

    fn source_component(i: &str) -> IResult<&str, &str, CustomError<&str>> {
        take_while1(is_source_char)(i)
    }

    #[allow(clippy::type_complexity)]
    fn id_range_bracketed_affix(i: &str) -> IResult<&str, IdRangeComponent, CustomError<&str>> {
        map_res(
            (
                opt(digit1),
                delimited(
                    char('['),
                    cut(separated_list1(tag(","), Self::id_range_step)),
                    char(']'),
                ),
                opt(digit1),
            ),
            |(high, ranges, low)| {
                let low = low
                    .map(|s| s.parse::<u32>().map(|value| (s.len(), value)))
                    .transpose()?
                    .map(|(len, value)| IdRangeOffset::new(value, len as u32))
                    .transpose()?;

                let high = high
                    .map(|s| s.parse::<u32>().map(|value| (s.len(), value)))
                    .transpose()?
                    .map(|(len, value)| IdRangeOffset::new(value, len as u32))
                    .transpose()?;

                Ok(IdRangeComponent::IdRange((high, ranges, low)))
            },
        )
        .parse(i)
    }

    #[allow(clippy::type_complexity)]
    fn id_standalone(i: &str) -> IResult<&str, IdRangeComponent, CustomError<&str>> {
        map_res(
            digit1,
            |d: &str| -> Result<IdRangeComponent, NodeSetParseError> {
                let start = d.parse::<u32>()?;
                Ok(IdRangeComponent::Single(SingleId::new(
                    start,
                    d.len() as u32,
                )?))
            },
        )
        .parse(i)
    }

    fn id_range_step_rangeset(i: &str) -> IResult<&str, IdRangeComponent, CustomError<&str>> {
        map_res(
            Self::id_range_step,
            |idrs| -> Result<IdRangeComponent, NodeSetParseError> {
                Ok(IdRangeComponent::IdRange((None, vec![idrs], None)))
            },
        )
        .parse(i)
    }

    #[allow(clippy::type_complexity)]
    fn id_range_step(i: &str) -> IResult<&str, IdRangeStep, CustomError<&str>> {
        map_res(pair(digit1,
                         opt((
                                 tag("-"),
                                digit1,
                                opt(
                                    pair(
                                        tag("/"),
                                        digit1))))),

                        |s: (&str, Option<(&str, &str, Option<(&str, &str)>)>) | -> Result<IdRangeStep, NodeSetParseError> {
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
                                        Some((_, step)) => {(end, step.parse::<u32>()?)}
                                    }
                                }
                            };


                            Ok(IdRangeStep::new(start, end, step,  s.0.len().try_into()?)?)
                        }
                        ).parse(i)
    }

    fn is_padded(s: &str) -> bool {
        s.starts_with('0') && s != "0"
    }
}

#[derive(Debug, PartialEq)]
enum IdRangeComponent {
    Single(SingleId),
    IdRange(
        (
            Option<IdRangeOffset>,
            Vec<IdRangeStep>,
            Option<IdRangeOffset>,
        ),
    ),
}

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
    use crate::collections::config::DummySource;
    use itertools::Itertools;

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
                    pad: 1
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
                    pad: 1
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
                    pad: 1
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
                    pad: 2
                }
            )
        );
    }

    #[test]
    fn test_id_range_bracketed_affix() {
        assert_eq!(
            Parser::id_range_bracketed_affix("[2]").unwrap(),
            (
                "",
                IdRangeComponent::IdRange((
                    None,
                    vec![IdRangeStep {
                        start: 2,
                        end: 2,
                        step: 1,
                        pad: 1
                    }],
                    None
                ))
            )
        );

        assert_eq!(
            Parser::id_range_bracketed_affix("[2,3-4,5-67/8]").unwrap(),
            (
                "",
                IdRangeComponent::IdRange((
                    None,
                    vec![
                        IdRangeStep {
                            start: 2,
                            end: 2,
                            step: 1,
                            pad: 1
                        },
                        IdRangeStep {
                            start: 3,
                            end: 4,
                            step: 1,
                            pad: 1
                        },
                        IdRangeStep {
                            start: 5,
                            end: 67,
                            step: 8,
                            pad: 1
                        }
                    ],
                    None
                ))
            )
        );

        assert_eq!(
            Parser::id_range_bracketed_affix("12[2-9]").unwrap(),
            (
                "",
                (IdRangeComponent::IdRange((
                    Some(IdRangeOffset { value: 12, pad: 2 }),
                    vec![IdRangeStep {
                        start: 2,
                        end: 9,
                        step: 1,
                        pad: 1
                    },],
                    None
                )))
            )
        );

        assert_eq!(
            Parser::id_range_bracketed_affix("05[2-3]").unwrap(),
            (
                "",
                (IdRangeComponent::IdRange((
                    Some(IdRangeOffset { value: 5, pad: 2 }),
                    vec![IdRangeStep {
                        start: 2,
                        end: 3,
                        step: 1,
                        pad: 1
                    },],
                    None
                )))
            )
        );

        assert_eq!(
            Parser::id_range_bracketed_affix("[2-3]5").unwrap(),
            (
                "",
                (IdRangeComponent::IdRange((
                    None,
                    vec![IdRangeStep {
                        start: 2,
                        end: 3,
                        step: 1,
                        pad: 1
                    },],
                    Some(IdRangeOffset { value: 5, pad: 1 }),
                )))
            )
        );

        assert_eq!(
            Parser::id_range_bracketed_affix("[2-3]05").unwrap(),
            (
                "",
                (IdRangeComponent::IdRange((
                    None,
                    vec![IdRangeStep {
                        start: 2,
                        end: 3,
                        step: 1,
                        pad: 1
                    },],
                    Some(IdRangeOffset { value: 5, pad: 2 }),
                )))
            )
        );

        assert!(Parser::id_range_bracketed_affix("[2,]").is_err());
        assert!(Parser::id_range_bracketed_affix("[/8]").is_err());
        assert!(Parser::id_range_bracketed_affix("[34-]").is_err());
    }

    #[test]
    fn test_id_range_bracketed() {
        assert_eq!(
            Parser::id_range_bracketed_affix("[2]").unwrap(),
            (
                "",
                (IdRangeComponent::IdRange((
                    None,
                    vec![IdRangeStep {
                        start: 2,
                        end: 2,
                        step: 1,
                        pad: 1
                    }],
                    None
                )))
            )
        );
        assert_eq!(
            Parser::id_range_bracketed_affix("[2,3-4,5-67/8]").unwrap(),
            (
                "",
                (IdRangeComponent::IdRange((
                    None,
                    vec![
                        IdRangeStep {
                            start: 2,
                            end: 2,
                            step: 1,
                            pad: 1
                        },
                        IdRangeStep {
                            start: 3,
                            end: 4,
                            step: 1,
                            pad: 1
                        },
                        IdRangeStep {
                            start: 5,
                            end: 67,
                            step: 8,
                            pad: 1
                        }
                    ],
                    None
                )))
            )
        );

        assert!(Parser::id_range_bracketed_affix("[2,]").is_err());
        assert!(Parser::id_range_bracketed_affix("[/8]").is_err());
        assert!(Parser::id_range_bracketed_affix("[34-]").is_err());
    }

    #[test]
    fn test_node_component() {
        test_component(Parser::node_component);
        assert_eq!(
            Parser::node_component("http://a1a").unwrap(),
            ("1a", "http://a")
        );
    }

    #[test]
    fn test_source_component() {
        test_component(Parser::source_component);
        assert_eq!(
            Parser::source_component("http://a1a").unwrap(),
            ("://a1a", "http")
        );
    }

    fn test_component(parser: impl Fn(&str) -> IResult<&str, &str, CustomError<&str>>) {
        assert_eq!(parser("abcd efg").unwrap(), (" efg", "abcd"));
        assert!(parser(" abcdefg").is_err());
        assert_eq!(parser("a_b-c.d+j/2efg").unwrap(), ("2efg", "a_b-c.d+j/"));
        assert!(parser("0ef").is_err());
    }

    #[test]
    fn test_group() {
        let mut resolver = Resolver::default();
        let mut source = DummySource::new();
        source.add("group1", "a1, a2");
        source.add("group2", "a3, a4");
        source.add("group:1", "a4, a5");
        source.add("group:2", "a5, a6");
        source.add("group:suffix", "a7,a8");
        source.add("2", "a10,a11");
        source.add("3", "a12,a13");
        source.add("04", "a14,a15");
        resolver.add_sources(vec![("source".to_string(), source)]);

        let parser = Parser::with_resolver(&resolver, None);

        assert_eq!(
            parser
                .group::<crate::IdRangeList>("@source:group1")
                .unwrap()
                .1
                .iter()
                .join(","),
            "a1,a2"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>("@source:group[1,2]")
                .unwrap()
                .1
                .iter()
                .join(","),
            "a1,a2,a3,a4"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>("@source:group:1")
                .unwrap()
                .1
                .iter()
                .join(","),
            "a4,a5"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>("@source:group:[1,2]")
                .unwrap()
                .1
                .iter()
                .join(","),
            "a4,a5,a6"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>("@source:group:suffix")
                .unwrap()
                .1
                .iter()
                .join(","),
            "a7,a8"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>("@source:[2-3]")
                .unwrap()
                .1
                .iter()
                .join(","),
            "a10,a11,a12,a13"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>("@source:04")
                .unwrap()
                .1
                .iter()
                .join(","),
            "a14,a15"
        );
    }
}
