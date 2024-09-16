use auto_enums::auto_enum;
use clap::{Parser, Subcommand};
use eyre::{Context, Result};
use itertools::Itertools;
use nodeset::{IdRangeList, NodeSet, Resolver};
use std::io;
use std::io::Read;

#[derive(Parser)]
#[command(about = "Operations on set of nodes")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fold nodesets or individual nodes into a nodeset
    Fold {
        /// Nodesets to fold
        nodeset: Option<Vec<String>>,
    },
    /// List individual nodes in nodesets
    List {
        /// Nodesets to expand into a list
        nodeset: Option<Vec<String>>,
        /// Separator between nodes
        #[arg(short, default_value = " ")]
        separator: String,
    },
    /// Count nodes in nodesets
    Count {
        /// Nodesets to count
        nodeset: Option<Vec<String>>,
    },
    /// List groups of nodes
    Groups {
        /// List groups from all sources
        #[arg(short)]
        all_sources: bool,
        /// List groups from the specified source
        #[arg(short, conflicts_with("all_sources"))]
        source: Option<String>,
        /// Display group members
        #[arg(short)]
        members: bool,
        /// Display groups intersecting with provided nodesets
        nodeset: Option<Vec<String>>,
    },
    /// List group sources
    Sources {},
}

fn main() -> Result<()> {
    env_logger::init();
    Resolver::set_global(Resolver::from_config()?);
    use std::io::Write;
    let args = Cli::parse();
    match args.command {
        Commands::Fold { nodeset } => {
            let nodeset = nodeset_argument(nodeset)?;
            println!("{}", nodeset);
        }
        Commands::List { nodeset, separator } => {
            let nodeset = nodeset_argument(nodeset)?;
            let mut it = nodeset.iter();

            let mut lock = io::stdout().lock();

            if let Some(first) = it.next() {
                lock.write_all(first.as_bytes())?;
            }
            for node in it {
                lock.write_all(separator.as_bytes())?;
                lock.write_all(node.as_bytes())?;
            }

            println!();
        }
        Commands::Count { nodeset } => {
            let nodeset = nodeset_argument(nodeset)?;
            println!("{}", nodeset.len());
        }
        Commands::Groups {
            all_sources,
            members,
            source,
            nodeset,
        } => {
            let nodeset = if nodeset.is_some() {
                Some(nodeset_argument(nodeset)?)
            } else {
                None
            };
            group_cmd(all_sources, source, members, nodeset);
        }
        Commands::Sources {} => {
            let resolver = Resolver::get_global();
            for source in resolver.sources() {
                println!(
                    "{}{}",
                    source,
                    if source == resolver.default_source() {
                        " (default)"
                    } else {
                        ""
                    }
                );
            }
        }
    }

    Ok(())
}

#[auto_enum]
fn group_cmd(
    all: bool,
    default_source: Option<String>,
    display_members: bool,
    filter: Option<NodeSet>,
) {
    let resolver = Resolver::get_global();

    let all_groups;
    let groups;

    #[auto_enum(Iterator)]
    let iter = if all {
        all_groups = resolver
            .list_all_groups::<IdRangeList>()
            .collect::<Vec<_>>();
        all_groups.iter().flat_map(|(source, groups)| {
            let source = if *source == resolver.default_source() {
                None
            } else {
                Some(*source)
            };

            groups.iter().map(move |group| (source, group))
        })
    } else {
        groups = resolver.list_groups::<IdRangeList>(default_source.as_deref());
        groups
            .iter()
            .map(|group| (default_source.as_deref(), group))
    };

    let s = iter
        .filter_map(|(source, group)| {
            let mut members = resolver.resolve::<IdRangeList>(source, &group).ok()?;

            if let Some(filter) = &filter {
                members = members.intersection(filter);
                if members.is_empty() {
                    return None;
                }
            }

            let display_source = match &source {
                Some(s) => format!("{}:", s),
                None => "".to_string(),
            };
            if display_members {
                Some(format!("@{}{} {}", display_source, group, members))
            } else {
                Some(format!("@{}{}", display_source, group))
            }
        })
        .sorted()
        .join("\n");

    println!("{}", s);
}

fn nodeset_argument(ns: Option<Vec<String>>) -> Result<NodeSet> {
    let nodeset: NodeSet = match ns {
        Some(v) if v == vec!["-".to_string()] => read_stdin()?,
        Some(v) => v.join(" "),
        None => read_stdin()?,
    }
    .parse()
    .context("failed to parse nodeset")?;

    Ok(nodeset)
}

fn read_stdin() -> Result<String> {
    let mut s = String::new();
    io::stdin()
        .lock()
        .read_to_string(&mut s)
        .context("failed to read standard input")?;
    Ok(s)
}
