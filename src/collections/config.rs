use super::nodeset::ConfigurationError;
use super::parsers::Parser;
use super::NodeSet;
use crate::idrange::IdRange;
use crate::NodeSetParseError;
use ini::Properties;
use log::debug;
use serde::Deserialize;
use shellexpand::env_with_context_no_errors;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

/// The default resolver used to parse NodeSet using the FromStr trait
static GLOBAL_RESOLVER: OnceLock<Resolver> = OnceLock::new();

/// Default group configuration paths
static CONFIG_PATHS: &[&str] = &[
    "$HOME/.local/etc/clustershell",
    "/etc/clustershell",
    "$XDG_CONFIG_HOME/clustershell",
];

/// An inventory of group sources used to resolve group names to node sets
///
/// The FromStr implementation of NodeSet uses the global resolver which can be
/// setup to read group sources from the default configuration file as follows:
///
/// ```rust,no_run
/// use nodeset::{NodeSet, Resolver};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     Resolver::set_global(Resolver::from_config()?).unwrap();
///
///     let ns: NodeSet = "@group".parse()?;
///
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct Resolver {
    sources: HashMap<String, Box<dyn GroupSource>>,
    default_source: String,
}

impl Default for Resolver {
    fn default() -> Self {
        Self {
            sources: HashMap::default(),
            default_source: "local".to_string(),
        }
    }
}

impl Resolver {
    /// Create a new resolver from the default configuration files
    pub fn from_config() -> Result<Self, ConfigurationError> {
        let mut group_config = MainGroupConfig::default();

        let mut cfg_dir = None;
        for &path in CONFIG_PATHS {
            if let Some(file) = open_config_path(&Path::new(&path).join("groups.conf")) {
                group_config.merge(MainGroupConfig::from_reader(BufReader::new(file))?);
                cfg_dir = resolve_config_path(Path::new(&path));
            }
        }

        if let Some(cfg_dir) = cfg_dir {
            if let Some(cfg_dir) = cfg_dir.to_str() {
                group_config.set_cfgdir(cfg_dir)?;
            }
        }

        Resolver::from_dynamic_config(group_config)
    }

    /// Create a new resolver from a dynamic group configuration
    ///
    /// `set_cfgdir` must already have been called on the dynamic group configuration
    fn from_dynamic_config(groups: MainGroupConfig) -> Result<Self, ConfigurationError> {
        let mut resolver = Resolver {
            sources: Default::default(),
            default_source: groups
                .config
                .as_ref()
                .and_then(|c| c.default.clone())
                .unwrap_or_else(|| "default".to_string()),
        };

        for autodir in groups.autodirs() {
            for path in find_files_with_ext(Path::new(&autodir), "yaml") {
                if let Some(file) = open_config_path(&path) {
                    let static_groups = StaticGroupConfig::from_reader(BufReader::new(file))?;
                    resolver.add_sources(static_groups);
                }
            }
        }
        for confdir in groups.confdirs() {
            for path in find_files_with_ext(Path::new(&confdir), "conf") {
                if let Some(file) = open_config_path(&path) {
                    let dynamic_groups = MainGroupConfig::from_reader(BufReader::new(file))?;
                    resolver.add_sources(dynamic_groups);
                }
            }
        }

        resolver.add_sources(groups);

        Ok(resolver)
    }

    /// Set the global resolver to use for parsing NodeSet using the FromStr trait
    ///
    /// Returns an error if the global resolver is already set
    pub fn set_global(resolver: Resolver) -> Result<(), Resolver> {
        GLOBAL_RESOLVER.set(resolver)?;

        Ok(())
    }

    /// Get the global resolver
    pub fn get_global() -> &'static Resolver {
        static DEFAULT_RESOLVER: OnceLock<Resolver> = OnceLock::new();

        GLOBAL_RESOLVER
            .get()
            .unwrap_or(DEFAULT_RESOLVER.get_or_init(Resolver::default))
    }

    /// Resolve a group name to a NodeSet
    ///
    /// If `source` is None, the default group source of the resolver is used.
    pub fn resolve<T: IdRange + PartialEq + Clone + Display + Debug>(
        &self,
        source: Option<&str>,
        group: &str,
    ) -> Result<NodeSet<T>, NodeSetParseError> {
        let source = source.unwrap_or(self.default_source.as_str());

        Parser::with_resolver(self, Some(source)).parse(
            &self
                .sources
                .get(source)
                .ok_or_else(|| NodeSetParseError::Source(source.to_owned()))?
                .map(group)?
                .unwrap_or_default(),
        )
    }

    /// List groups from a source
    ///
    /// If `source` is None, the default group source of the resolver is used.
    pub fn list_groups<T: IdRange + PartialEq + Clone + Display + Debug>(
        &self,
        source: Option<&str>,
    ) -> NodeSet<T> {
        let source = source.unwrap_or(self.default_source.as_str());

        Parser::default()
            .parse(
                &self
                    .sources
                    .get(source)
                    .map(|s| s.list())
                    .unwrap_or_default(),
            )
            .unwrap_or_default()
    }

    /// List groups from all sources
    ///
    /// Returns a list of tuples with the source name and the group name
    pub fn list_all_groups<T: IdRange + PartialEq + Clone + Display + Debug>(
        &self,
    ) -> impl Iterator<Item = (&str, NodeSet<T>)> {
        self.sources.iter().map(|(source, groups)| {
            (
                source.as_str(),
                Parser::default().parse(&groups.list()).unwrap_or_default(),
            )
        })
    }

    /// List all sources
    pub fn sources(&self) -> impl Iterator<Item = &String> {
        self.sources.keys()
    }

    /// Returns the default group source for this resolver
    pub fn default_source(&self) -> &str {
        &self.default_source
    }

    pub(crate) fn add_sources(
        &mut self,
        sources: impl IntoIterator<Item = (String, impl GroupSource + 'static)>,
    ) {
        sources.into_iter().for_each(|(name, source)| {
            self.sources.insert(name, Box::new(source));
        });
    }
}

/// Open a config file from a path, expanding environment variables.
///
/// Returns None if there was any failure
fn open_config_path(path: &Path) -> Option<std::fs::File> {
    resolve_config_path(path).and_then(|p| std::fs::File::open(p).ok())
}

/// Expands environment variables in a path
///
/// Returns None in case of non-utf8 path
fn resolve_config_path(path: &Path) -> Option<PathBuf> {
    let context = |s: &str| match s {
        "HOME" => std::env::var("HOME").ok(),
        "XDG_CONFIG_HOME" => std::env::var("XDG_CONFIG_HOME").ok().or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| Path::new(&h).join(".config").to_str().unwrap().to_string())
        }),
        _ => None,
    };

    Some(PathBuf::from(
        env_with_context_no_errors(path.to_str()?, context).as_ref(),
    ))
}

/// Returns a list of files with a given extension in a directory.
///
/// Returns an empty list in case of any failure
fn find_files_with_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut files = vec![];

    let Ok(it) = fs::read_dir(dir) else {
        return files;
    };

    for entry in it {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_file() && path.extension().map(|ext| ext.to_str()) == Some(Some(ext)) {
            files.push(path);
        }
    }

    files
}

/// Trait for group resolution features of a group source
pub(crate) trait GroupSource: Debug + Send + Sync {
    fn map(&self, group: &str) -> Result<Option<String>, NodeSetParseError>;
    fn list(&self) -> String;
}

/// Settings from the main group configuration file (groups.conf)
#[derive(Debug, Default)]
struct MainGroupConfig {
    config: Option<ResolverOptions>,
    sources: HashMap<String, DynamicGroupSource>,
}

impl MainGroupConfig {
    fn from_reader(mut reader: impl std::io::Read) -> Result<Self, ConfigurationError> {
        use ini::Ini;

        let parser = Ini::read_from_noescape(&mut reader)?;
        let mut config = MainGroupConfig::default();
        for (sec, prop) in parser.iter() {
            match sec {
                Some("Main") => {
                    config.config = Some(prop.try_into()?);
                }
                Some(sources) => {
                    for source in sources.split(',') {
                        config.sources.insert(
                            source.to_string(),
                            DynamicGroupSource::from_props(prop, source.to_string())?,
                        );
                    }
                }
                None => {
                    if let Some(key) = prop.iter().next().map(|(k, _)| k) {
                        return Err(ConfigurationError::UnexpectedProperty(key.to_string()));
                    }
                }
            }
        }

        Ok(config)
    }

    fn autodirs(&self) -> Vec<String> {
        self.config
            .as_ref()
            .map(|c| c.autodirs())
            .unwrap_or_default()
    }

    fn confdirs(&self) -> Vec<String> {
        self.config
            .as_ref()
            .map(|c| c.confdirs())
            .unwrap_or_default()
    }

    fn set_cfgdir(&mut self, cfgdir: &str) -> Result<(), ConfigurationError> {
        let context = |s: &str| match s {
            "CFGDIR" => Some(cfgdir),
            _ => None,
        };

        if let Some(config) = &mut self.config {
            config.confdir = config
                .confdir
                .as_ref()
                .map(|c| env_with_context_no_errors(&c, context).to_string());
            config.autodir = config
                .autodir
                .as_ref()
                .map(|c| env_with_context_no_errors(&c, context).to_string());
        }

        for (_, group) in self.sources.iter_mut() {
            group.set_cfgdir(cfgdir)?;
        }

        Ok(())
    }

    /// Merge settings for another group configuration file into this one
    fn merge(&mut self, other: Self) {
        match (&mut self.config, other.config) {
            (Some(ref mut main), Some(other_main)) => main.merge(other_main),
            (None, Some(other_main)) => self.config = Some(other_main),
            _ => (),
        }
        self.sources.extend(other.sources);
    }
}

impl IntoIterator for MainGroupConfig {
    type Item = (String, DynamicGroupSource);
    type IntoIter = std::collections::hash_map::IntoIter<String, DynamicGroupSource>;

    fn into_iter(self) -> Self::IntoIter {
        self.sources.into_iter()
    }
}

#[derive(Debug, Default)]
struct ResolverOptions {
    default: Option<String>,
    confdir: Option<String>,
    autodir: Option<String>,
}

impl ResolverOptions {
    /// Merge resolver options from `other` into `self`
    fn merge(&mut self, other: Self) {
        if let Some(default) = other.default {
            self.default = Some(default);
        }
        if let Some(confdir) = other.confdir {
            self.confdir = Some(confdir);
        }
        if let Some(autodir) = other.autodir {
            self.autodir = Some(autodir);
        }
    }

    fn autodirs(&self) -> Vec<String> {
        self.autodir
            .as_ref()
            .and_then(|autodir| shlex::split(autodir))
            .unwrap_or_default()
    }

    fn confdirs(&self) -> Vec<String> {
        self.confdir
            .as_ref()
            .and_then(|confdir| shlex::split(confdir))
            .unwrap_or_default()
    }
}

impl TryFrom<&Properties> for ResolverOptions {
    type Error = ConfigurationError;

    fn try_from(props: &Properties) -> Result<Self, Self::Error> {
        let mut res = Self::default();

        for (k, v) in props.iter() {
            match k {
                "default" => {
                    res.default = Some(v.to_string());
                }
                "confdir" => {
                    res.confdir = Some(v.to_string());
                }
                "autodir" => {
                    res.autodir = Some(v.to_string());
                }
                _ => {
                    return Err(ConfigurationError::UnexpectedProperty(k.to_string()));
                }
            }
        }

        Ok(res)
    }
}

/// Settings from a dynamic group source (groups.conf.d/<source>.conf)
#[derive(Debug)]
struct DynamicGroupSource {
    name: String,
    map: String,
    all: Option<String>,
    list: Option<String>,
}

impl DynamicGroupSource {
    fn from_props(props: &Properties, name: String) -> Result<Self, ConfigurationError> {
        let map = props
            .get("map")
            .ok_or_else(|| ConfigurationError::MissingProperty("map".to_string()))?
            .to_string();
        let all = props.get("all").map(|s| s.to_string());
        let list = props.get("list").map(|s| s.to_string());

        Ok(Self {
            name,
            map,
            all,
            list,
        })
    }

    fn set_cfgdir(&mut self, cfgdir: &str) -> Result<(), ConfigurationError> {
        let context = |s: &str| match s {
            "CFGDIR" => Some(cfgdir),
            "SOURCE" => Some(self.name.as_str()),
            _ => None,
        };

        self.map = env_with_context_no_errors(&self.map, context).to_string();
        self.all = self
            .all
            .as_ref()
            .map(|s| env_with_context_no_errors(s, context).to_string());
        self.list = self
            .list
            .as_ref()
            .map(|s| env_with_context_no_errors(s, context).to_string());

        Ok(())
    }
}

impl GroupSource for DynamicGroupSource {
    fn map(&self, group: &str) -> Result<Option<String>, NodeSetParseError> {
        let context = |s: &str| match s {
            "GROUP" => Some(group),
            "SOURCE" => Some(self.name.as_str()),
            _ => None,
        };
        let map = env_with_context_no_errors(&self.map, context).to_string();

        let output = Command::new("/bin/sh").arg("-c").arg(&map).output()?;

        if !output.status.success() {
            return Err(NodeSetParseError::Command(std::io::Error::other(format!(
                "Command '{}' returned non-zero exit code",
                map
            ))));
        }

        let res = String::from_utf8_lossy(&output.stdout);

        debug!(
            "Map command '{}' for @'{}':'{}' returned: {}",
            map, self.name, group, res
        );

        Ok(Some(res.trim().to_string()))
    }

    fn list(&self) -> String {
        let Some(ref list_cmd) = self.list else {
            return Default::default();
        };

        let context = |s: &str| match s {
            "SOURCE" => Some(self.name.as_str()),
            _ => None,
        };
        let list = env_with_context_no_errors(&list_cmd, context).to_string();

        let output = Command::new("/bin/sh")
            .arg("-c")
            .arg(&list)
            .output()
            .unwrap();

        if !output.status.success() {
            panic!("Command '{}' returned non-zero exit code", list);
        }

        let res = String::from_utf8_lossy(&output.stdout);

        debug!(
            "List command '{}' for @'{}':* returned: {}",
            list, self.name, res
        );

        res.trim().to_string()
    }
}

/// Settings from a static group source configuration file (groups.d/*.yaml)
#[derive(Deserialize, Debug)]
struct StaticGroupConfig {
    #[serde(flatten)]
    sources: HashMap<String, StaticGroupSource>,
}

impl StaticGroupConfig {
    fn from_reader(reader: impl std::io::Read) -> Result<Self, ConfigurationError> {
        let config: Self = serde_yaml::from_reader(reader)?;
        Ok(config)
    }
}

impl IntoIterator for StaticGroupConfig {
    type Item = (String, StaticGroupSource);
    type IntoIter = std::collections::hash_map::IntoIter<String, StaticGroupSource>;

    fn into_iter(self) -> Self::IntoIter {
        self.sources.into_iter()
    }
}

#[derive(Deserialize, Debug)]
struct StaticGroupSource {
    #[serde(flatten)]
    groups: HashMap<String, SingleOrVec>,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum SingleOrVec {
    Single(String),
    Vec(Vec<String>),
}

impl From<&SingleOrVec> for String {
    fn from(s: &SingleOrVec) -> Self {
        match s {
            SingleOrVec::Single(s) => s.clone(),
            SingleOrVec::Vec(v) => v.join(","),
        }
    }
}

impl GroupSource for StaticGroupSource {
    fn map(&self, group: &str) -> Result<Option<String>, NodeSetParseError> {
        Ok(self.groups.get(group).map(|v| v.into()))
    }

    fn list(&self) -> String {
        use itertools::Itertools;
        self.groups.keys().join(" ")
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct DummySource {
    map: HashMap<String, String>,
}
#[cfg(test)]
impl DummySource {
    pub(crate) fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub(crate) fn add(&mut self, group: &str, nodes: &str) {
        self.map.insert(group.to_string(), nodes.to_string());
    }
}

#[cfg(test)]
impl GroupSource for DummySource {
    fn map(&self, group: &str) -> Result<Option<String>, NodeSetParseError> {
        Ok(self.map.get(group).cloned())
    }

    fn list(&self) -> String {
        use itertools::Itertools;

        self.map.keys().join(" ")
    }
}

#[cfg(test)]
mod tests {
    use crate::{collections::parsers::Parser, IdRangeList};

    use super::*;

    #[test]
    fn test_static_config() {
        let config = include_str!("tests/cluster.yaml");
        let mut resolver = Resolver::default();
        resolver.add_sources(StaticGroupConfig::from_reader(config.as_bytes()).unwrap());
        let parser = Parser::with_resolver(&resolver, Some("roles"));
        assert_eq!(
            parser.parse::<IdRangeList>("@login").unwrap().to_string(),
            "login[1-2]"
        );

        assert_eq!(
            parser.parse::<IdRangeList>("@*").unwrap(),
            parser
                .parse::<IdRangeList>(
                    "node[0001-0288],mds[1-4],oss[0-15],server0001,login[1-2],mgmt[1-2]"
                )
                .unwrap()
        );

        match parser.parse::<IdRangeList>("@login:aa") {
            Err(NodeSetParseError::Source(_)) => (),
            e => panic!("Expected Source error, got {e:?}",),
        }
        assert_eq!(
            parser
                .parse::<IdRangeList>("@roles:cpu_only")
                .unwrap()
                .to_string(),
            "node[0009-0288]"
        );

        assert_eq!(
            parser
                .parse::<IdRangeList>("@roles:non_existent")
                .unwrap()
                .to_string(),
            ""
        );

        match parser.parse::<IdRangeList>("@non_existent:non_existent") {
            Err(NodeSetParseError::Source(_)) => (),
            _ => panic!("Expected Source error"),
        }

        let ns1 = parser.parse::<IdRangeList>("@rack[1-2]:hsw").unwrap();
        let ns2 = "mgmt[1-2],oss[0-15],mds[1-4]".parse().unwrap();
        assert_eq!(ns1, ns2);

        let ns1 = parser.parse::<IdRangeList>("@rack[1-2]:*").unwrap();
        let ns2 = "mgmt[1-2],oss[0-15],mds[1-4],node[0001-0288]"
            .parse()
            .unwrap();
        assert_eq!(ns1, ns2);

        let ns1 = parser.parse::<IdRangeList>("@network:net[1,3]").unwrap();
        let ns2 = "node[10-19,30-39]".parse().unwrap();
        assert_eq!(ns1, ns2);

        assert_eq!(
            resolver.list_groups::<IdRangeList>(Some("numerical")),
            "1-2,03".parse::<NodeSet>().unwrap()
        );

        assert_eq!(
            resolver
                .resolve::<IdRangeList>(Some("numerical"), "1")
                .unwrap(),
            "node[10-19]".parse::<NodeSet>().unwrap()
        );
    }

    #[test]
    fn test_parse_dynamic_config() {
        use tempfile::TempDir;

        let config = include_str!("tests/groups.conf");
        let mut dynamic = MainGroupConfig::from_reader(config.as_bytes()).unwrap();

        let tmp_dir = TempDir::new().unwrap();

        std::fs::create_dir(tmp_dir.path().join("groups.d")).unwrap();
        std::fs::write(
            tmp_dir.path().join("groups.d").join("local.cfg"),
            include_str!("tests/local.cfg"),
        )
        .unwrap();

        std::fs::create_dir(tmp_dir.path().join("groups.conf.d")).unwrap();
        std::fs::write(
            tmp_dir.path().join("groups.conf.d").join("multi.conf"),
            include_str!("tests/multi.conf"),
        )
        .unwrap();

        dynamic
            .set_cfgdir(tmp_dir.path().to_str().unwrap())
            .unwrap();

        assert_eq!(
            dynamic.autodirs(),
            vec![
                "/etc/clustershell/groups.d",
                &format!("{}/groups.d", tmp_dir.path().to_str().unwrap())
            ]
        );
        assert_eq!(
            dynamic.confdirs(),
            vec![
                "/etc/clustershell/groups.conf.d",
                &format!("{}/groups.conf.d", tmp_dir.path().to_str().unwrap())
            ]
        );

        let resolver = Resolver::from_dynamic_config(dynamic).unwrap();

        assert_eq!(
            resolver
                .resolve::<IdRangeList>(None, "oss")
                .unwrap()
                .to_string(),
            "example[4-5]"
        );

        assert_eq!(
            resolver
                .resolve::<IdRangeList>(Some("local"), "mds")
                .unwrap()
                .to_string(),
            "example6"
        );

        assert_eq!(
            resolver.list_groups::<IdRangeList>(Some("local")),
            "compute,gpu,all,adm,io,mds,oss,[1-2],03"
                .parse::<NodeSet>()
                .unwrap()
        );

        assert_eq!(
            resolver.resolve::<IdRangeList>(Some("local"), "1").unwrap(),
            "example[32-33]".parse::<NodeSet>().unwrap()
        );

        assert_eq!(
            resolver.list_groups::<IdRangeList>(Some("rack1")),
            "rack1_switches[1-4],rack1_nodes[1-4]"
                .parse::<NodeSet>()
                .unwrap()
        );

        assert_eq!(
            resolver.list_groups::<IdRangeList>(Some("rack2")),
            "rack2_switches[1-4],rack2_nodes[1-4]"
                .parse::<NodeSet>()
                .unwrap()
        );

        assert_eq!(
            resolver
                .resolve::<IdRangeList>(Some("rack1"), "nodes")
                .unwrap(),
            "rack1_nodes[1-4]".parse::<NodeSet>().unwrap()
        );
    }
}
