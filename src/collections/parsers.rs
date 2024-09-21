use super::{config::Resolver, nodeset::NodeSetDimensions};
use crate::{
    collections::{idset::IdRangeProduct, nodeset::IdSetKind},
    idrange::{AffixIdRangeStep, IdRange, IdRangeList, IdRangeOffset, IdRangeStep, SingleId},
    IdSet, NodeSet, NodeSetParseError,
};
use auto_enums::auto_enum;
use std::{convert::TryInto, fmt};
use winnow::{
    self,
    ascii::{digit1, multispace0, multispace1},
    combinator::{
        alt, cut_err, delimited, eof, opt, peek, preceded, repeat, separated, separated_pair,
        terminated,
    },
    error::{
        ErrMode, FromExternalError, ModalResult as GenericModalResult, ParseError, ParserError,
    },
    token::{literal, one_of, take_while},
    Parser as WinParser,
};

type ModalResult<T> = GenericModalResult<T, NodeSetParseError>;

impl ParserError<&str> for NodeSetParseError {
    type Inner = Self;

    fn from_input(input: &&str) -> Self {
        NodeSetParseError::Generic(input.to_string())
    }

    fn into_inner(self) -> Result<Self::Inner, Self> {
        Ok(self)
    }
}

impl From<ParseError<&str, NodeSetParseError>> for NodeSetParseError {
    fn from(e: ParseError<&str, NodeSetParseError>) -> Self {
        e.into_inner()
    }
}

impl<E> FromExternalError<&str, E> for NodeSetParseError
where
    E: Into<NodeSetParseError>,
{
    fn from_external_error(_: &&str, error: E) -> Self {
        error.into()
    }
}

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
        let mut ns = self.expr().parse(i)?;

        ns.fold();
        Ok(ns)
    }

    fn expr<T>(self) -> impl 'a + FnMut(&mut &str) -> ModalResult<NodeSet<T>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        move |i: &mut &str| {
            delimited(
                multispace0,
                repeat(0.., (self.term(), opt(Self::op)))
                    .fold(
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
                    )
                    .map(|(ns, _, _err)| ns.unwrap_or_default()),
                multispace0,
            )
            .parse_next(i)
        }
    }

    fn term<T>(self) -> impl 'a + FnMut(&mut &str) -> ModalResult<NodeSet<T>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        move |input: &mut &str| {
            alt((self.group_or_nodeset(), delimited("(", self.expr(), ")"))).parse_next(input)
        }
    }

    fn op(i: &mut &str) -> ModalResult<char> {
        alt((
            delimited(multispace0, one_of([',', '&', '!', '^']), multispace0),
            delimited(multispace1, one_of(['-']), multispace1),
            multispace1.value(' '),
        ))
        .parse_next(i)
    }

    fn group_or_nodeset<T>(self) -> impl 'a + FnMut(&mut &str) -> ModalResult<NodeSet<T>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        move |input: &mut &str| alt((Self::rangeset, Self::nodeset, self.group())).parse_next(input)
    }

    fn nodeset_or_rangeset<T>(i: &mut &str) -> ModalResult<NodeSet<T>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        alt((Self::nodeset, Self::rangeset)).parse_next(i)
    }

    fn rangeset<T>(i: &mut &str) -> ModalResult<NodeSet<T>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        (
            alt((Self::id_range_bracketed_affix, Self::id_range_step_rangeset)),
            peek(alt((",", "&", "!", "^", "(", ")", multispace1, eof))),
        )
            .try_map(|(idrs, _)| {
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
                Result::<_, NodeSetParseError>::Ok(ns)
            })
            .parse_next(i)
    }

    fn nodeset<T>(i: &mut &str) -> ModalResult<NodeSet<T>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        Self::set(false).parse_next(i)
    }

    fn sourceset<T>(i: &mut &str) -> ModalResult<NodeSet<T>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        Self::set(true).parse_next(i)
    }

    fn source_or_node_component<'k>(
        source: bool,
    ) -> impl 'a + FnMut(&mut &'k str) -> ModalResult<&'k str> {
        move |i: &mut &'k str| {
            if source {
                take_while(1.., is_source_char).parse_next(i)
            } else {
                take_while(1.., is_nodeset_char).parse_next(i)
            }
        }
    }

    fn set<T>(source: bool) -> impl 'a + FnMut(&mut &str) -> ModalResult<NodeSet<T>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        move |i: &mut &str| {
            (
                opt(alt((Self::id_range_bracketed_affix, Self::id_standalone))),
                repeat(
                    0..,
                    (
                        Self::source_or_node_component(source),
                        alt((Self::id_range_bracketed_affix, Self::id_standalone)),
                    ),
                ),
                opt(Self::source_or_node_component(source)),
            )
                .verify(|(_, components, suffix): &(_, Vec<_>, _)| {
                    // This is a rangeset
                    !(components.is_empty() && suffix.is_none())
                })
                .try_map(|(prefix, components, suffix)| {
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

                    Result::<_, NodeSetParseError>::Ok(ns)
                })
                .parse_next(i)
        }
    }

    #[auto_enum]
    fn group<T>(self) -> impl 'a + FnMut(&mut &str) -> ModalResult<NodeSet<T>>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        move |i: &mut &str| {
            preceded(
                "@",
                // Match either 'sources:groups' or 'groups'
                // Both sources and groups can be sets i.e: @source[1-4]:group[1,5]
                alt((
                    literal("*").value((None, None)),
                    self.group_with_source(),
                    Self::nodeset_or_rangeset.map(|s: NodeSet<IdRangeList>| (None, Some(s))),
                )),
            )
            .map(
                |(sources, groups)| -> Result<NodeSet<T>, ErrMode<NodeSetParseError>> {
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
                            let nodeset = resolver
                                .resolve(source.as_deref(), &group)
                                .map_err(ErrMode::Cut)?;
                            ns.extend_from_nodeset(&nodeset);
                        }
                    }

                    Ok(ns)
                },
            )
            .parse_next(i)?
        }
    }

    fn group_with_source<T>(
        self,
    ) -> impl FnMut(&mut &str) -> ModalResult<(Option<NodeSet<T>>, Option<NodeSet<T>>)>
    where
        T: IdRange + PartialEq + Clone + fmt::Display + fmt::Debug,
    {
        move |i: &mut &str| {
            alt((
                terminated(Self::sourceset, ":*").map(|source| (Some(source), None)),
                separated_pair(Self::sourceset, ":", opt(Self::nodeset_or_rangeset))
                    .map(|source| (Some(source.0), Some(source.1.unwrap_or_default()))),
            ))
            .parse_next(i)
        }
    }

    fn id_range_bracketed_affix(i: &mut &str) -> ModalResult<IdRangeComponent> {
        (
            opt(digit1),
            delimited("[", separated(1.., cut_err(Self::id_range_step), ","), "]"),
            opt(digit1),
        )
            .try_map(|(high, ranges, low)| {
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

                Result::<_, NodeSetParseError>::Ok(IdRangeComponent::IdRange((high, ranges, low)))
            })
            .parse_next(i)
    }

    fn id_standalone(i: &mut &str) -> ModalResult<IdRangeComponent> {
        digit1
            .try_map(|d: &str| -> Result<IdRangeComponent, NodeSetParseError> {
                let start = d.parse::<u32>()?;
                Ok(IdRangeComponent::Single(SingleId::new(
                    start,
                    d.len() as u32,
                )?))
            })
            .parse_next(i)
    }

    fn id_range_step_rangeset(i: &mut &str) -> ModalResult<IdRangeComponent> {
        Self::id_range_step
            .map(|idrs| IdRangeComponent::IdRange((None, vec![idrs], None)))
            .parse_next(i)
    }

    fn id_range_step(i: &mut &str) -> ModalResult<IdRangeStep> {
        (digit1, opt(("-", digit1, opt(("/", digit1))))).try_map(
            |s: (&str, Option<(&str, &str, Option<(&str, &str)>)>)| -> Result<IdRangeStep, NodeSetParseError> {
                let start = s.0.parse::<u32>()?;
                let mut padded = Self::is_padded(s.0);

                let (end, step) = match s.1 {
                    None => {(start, 1)},
                    Some(s1) => {
                        padded |= Self::is_padded(s1.1);

                        if padded && s1.1.len() != s.0.len() {
                            return Err(NodeSetParseError::MismatchedPadding(s.0.to_string(), s1.1.to_string()));
                        }

                        let end = s1.1.parse::<u32>()?;
                        match s1.2 {
                            None => {(end, 1)},
                            Some((_, step)) => {(end, step.parse::<u32>()?)}
                        }
                    }
                };


                Ok(IdRangeStep::new(start, end, step,  s.0.len().try_into()?)?)
            }
        ).parse_next(i)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections::config::DummySource;
    use itertools::Itertools;

    #[test]
    fn test_id_range_step() {
        assert_eq!(
            Parser::id_range_step(&mut "2").unwrap(),
            IdRangeStep {
                start: 2,
                end: 2,
                step: 1,
                pad: 1
            }
        );
        assert_eq!(
            Parser::id_range_step(&mut "2-34").unwrap(),
            IdRangeStep {
                start: 2,
                end: 34,
                step: 1,
                pad: 1
            }
        );
        assert_eq!(
            Parser::id_range_step(&mut "2-34/8").unwrap(),
            IdRangeStep {
                start: 2,
                end: 34,
                step: 8,
                pad: 1
            }
        );

        assert!(Parser::id_range_step(&mut "-34/8").is_err());
        assert!(Parser::id_range_step(&mut "/8").is_err());
        assert_eq!(
            Parser::id_range_step(&mut "34/8").unwrap(),
            IdRangeStep {
                start: 34,
                end: 34,
                step: 1,
                pad: 2
            }
        );
    }

    #[test]
    fn test_id_range_bracketed_affix() {
        assert_eq!(
            Parser::id_range_bracketed_affix(&mut "[2]").unwrap(),
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
        );

        assert_eq!(
            Parser::id_range_bracketed_affix(&mut "[2,3-4,5-67/8]").unwrap(),
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
        );

        assert_eq!(
            Parser::id_range_bracketed_affix(&mut "12[2-9]").unwrap(),
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
        );

        assert_eq!(
            Parser::id_range_bracketed_affix(&mut "05[2-3]").unwrap(),
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
        );

        assert_eq!(
            Parser::id_range_bracketed_affix(&mut "[2-3]5").unwrap(),
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
        );

        assert_eq!(
            Parser::id_range_bracketed_affix(&mut "[2-3]05").unwrap(),
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
        );

        assert!(Parser::id_range_bracketed_affix(&mut "[2,]").is_err());
        assert!(Parser::id_range_bracketed_affix(&mut "[/8]").is_err());
        assert!(Parser::id_range_bracketed_affix(&mut "[34-]").is_err());
    }

    #[test]
    fn test_id_range_bracketed() {
        assert_eq!(
            Parser::id_range_bracketed_affix(&mut "[2]").unwrap(),
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
        );
        assert_eq!(
            Parser::id_range_bracketed_affix(&mut "[2,3-4,5-67/8]").unwrap(),
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
        );

        assert!(Parser::id_range_bracketed_affix(&mut "[2,]").is_err());
        assert!(Parser::id_range_bracketed_affix(&mut "[/8]").is_err());
        assert!(Parser::id_range_bracketed_affix(&mut "[34-]").is_err());
    }

    #[test]
    fn test_node_component() {
        //test_component(Parser::node_component);
        assert_eq!(
            Parser::source_or_node_component(false)
                .parse_next(&mut "http://a1a")
                .unwrap(),
            "http://a"
        );
    }

    #[test]
    fn test_source_component() {
        //test_component(Parser::source_component);
        assert_eq!(
            Parser::source_or_node_component(true)
                .parse_next(&mut "http://a1a")
                .unwrap(),
            "http"
        );
    }

    //fn test_component<'a>(parser: impl Fn(&mut &'a str) -> ModalResult<&'a str>) {
    //    assert_eq!(parser(&mut "abcd efg").unwrap(), "abcd");
    //    assert!(parser(&mut " abcdefg").is_err());
    //    assert_eq!(parser(&mut "a_b-c.d+j/2efg").unwrap(), "a_b-c.d+j/");
    //    assert!(parser(&mut "0ef").is_err());
    //}

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
                .group::<crate::IdRangeList>()
                .parse(&mut "@source:group1")
                .unwrap()
                .iter()
                .join(","),
            "a1,a2"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>()
                .parse(&mut "@source:group[1,2]")
                .unwrap()
                .iter()
                .join(","),
            "a1,a2,a3,a4"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>()
                .parse(&mut "@source:group:1")
                .unwrap()
                .iter()
                .join(","),
            "a4,a5"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>()
                .parse(&mut "@source:group:[1,2]")
                .unwrap()
                .iter()
                .join(","),
            "a4,a5,a6"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>()
                .parse(&mut "@source:group:suffix")
                .unwrap()
                .iter()
                .join(","),
            "a7,a8"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>()
                .parse(&mut "@source:[2-3]")
                .unwrap()
                .iter()
                .join(","),
            "a10,a11,a12,a13"
        );

        assert_eq!(
            parser
                .group::<crate::IdRangeList>()
                .parse(&mut "@source:04")
                .unwrap()
                .iter()
                .join(","),
            "a14,a15"
        );
    }
}
