use super::nodeset::ConfigurationError;
use super::parsers::Parser;
use super::NodeSet;
use crate::idrange::IdRange;
use crate::NodeSetParseError;
use ini::Properties;
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
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Deserialize, Debug)]
struct StaticGroupConfig {
    #[serde(flatten)]
    sources: HashMap<String, StaticGroupSource>,
}

/// Settings from a static group source configuration file (groups.d/*.yaml)
#[derive(Deserialize, Debug)]
struct StaticGroupSource {
    #[serde(flatten)]
    groups: HashMap<String, SingleOrVec>,
}

#[derive(Debug, Default)]
struct ResolverConfig {
    default: Option<String>,
    confdir: Option<String>,
    autodir: Option<String>,
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

impl ResolverConfig {
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

/// Settings from the dynamic/external group configuration file (groups.conf)
#[derive(Debug, Default)]
struct DynamicGroupConfig {
    config: Option<ResolverConfig>,
    groups: HashMap<String, DynamicGroupSource>,
}

/// Settings from a dynamic group source (groups.conf.d/<source>.conf)
#[derive(Debug)]
struct DynamicGroupSource {
    name: String,
    map: String,
    all: Option<String>,
    list: Option<String>,
}

trait GroupSource: Debug + Send + Sync {
    fn map(&self, group: &str) -> Result<Option<String>, NodeSetParseError>;
    fn list(&self) -> String;
}

impl TryFrom<&Properties> for ResolverConfig {
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

impl DynamicGroupConfig {
    fn from_reader(mut reader: impl std::io::Read) -> Result<Self, ConfigurationError> {
        use ini::Ini;

        let parser = Ini::read_from_noescape(&mut reader)?;
        let mut config = DynamicGroupConfig::default();
        for (sec, prop) in parser.iter() {
            match sec {
                Some("Main") => {
                    config.config = Some(prop.try_into()?);
                }
                Some(sources) => {
                    for source in sources.split(',') {
                        config.groups.insert(
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

        for (_, group) in self.groups.iter_mut() {
            group.set_cfgdir(cfgdir)?;
        }

        Ok(())
    }

    fn merge(&mut self, other: Self) {
        match (&mut self.config, other.config) {
            (Some(ref mut main), Some(other_main)) => main.merge(other_main),
            (None, Some(other_main)) => self.config = Some(other_main),
            _ => (),
        }
        self.groups.extend(other.groups);
    }
}

impl IntoIterator for DynamicGroupConfig {
    type Item = (String, DynamicGroupSource);
    type IntoIter = std::collections::hash_map::IntoIter<String, DynamicGroupSource>;

    fn into_iter(self) -> Self::IntoIter {
        self.groups.into_iter()
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
            return Err(NodeSetParseError::Command(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Command '{}' returned non-zero exit code", map),
            )));
        }
        Ok(Some(
            String::from_utf8_lossy(&output.stdout).trim().to_string(),
        ))
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

        String::from_utf8_lossy(&output.stdout).to_string()
    }
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

impl GroupSource for StaticGroupSource {
    fn map(&self, group: &str) -> Result<Option<String>, NodeSetParseError> {
        Ok(self.groups.get(group).map(|v| v.into()))
    }

    fn list(&self) -> String {
        use itertools::Itertools;
        self.groups.keys().join(" ")
    }
}

/// An inventory of group sources used to resolve group names to node sets
///
/// The FromStr implementation of NodeSet uses the global resolver which can be
/// setup to read group sources from the default configuration file as follows:
///
/// ```rust
/// use ns::{NodeSet, Resolver};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     Resolver::set_global(Resolver::from_config()?);
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

static GLOBAL_RESOLVER: RwLock<Option<Arc<Resolver>>> = RwLock::new(None);

impl Default for Resolver {
    fn default() -> Self {
        Self {
            sources: HashMap::default(),
            default_source: "local".to_string(),
        }
    }
}

static CONFIG_PATHS: &[&str] = &[
    "$HOME/.local/etc/clustershell",
    "/etc/clustershell",
    "$XDG_CONFIG_HOME/clustershell",
];

// Expand environment variables in a path
// Returns None in case of non-utf8 path
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

// Open a config file from a path, expanding environment variables
// Returns None if there was any failure (file not found, insufficient permissions, non-utf8 path, etc.)
fn open_config_path(path: &Path) -> Option<std::fs::File> {
    resolve_config_path(path).and_then(|p| std::fs::File::open(p).ok())
}

fn find_files_with_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut yaml_files = vec![];

    let Ok(it) = fs::read_dir(dir) else {
        return yaml_files;
    };

    for entry in it {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_file() && path.extension().map(|ext| ext.to_str()) == Some(Some(ext)) {
            yaml_files.push(path);
        }
    }

    yaml_files
}

impl Resolver {
    /// Create a new resolver from the default configuration files
    pub fn from_config() -> Result<Self, ConfigurationError> {
        let mut groups = DynamicGroupConfig::default();

        let mut cfg_dir = None;
        for &path in CONFIG_PATHS {
            if let Some(file) = open_config_path(&Path::new(&path).join("groups.conf")) {
                groups.merge(DynamicGroupConfig::from_reader(BufReader::new(file))?);
                cfg_dir = resolve_config_path(Path::new(&path));
            }
        }

        if let Some(cfg_dir) = cfg_dir {
            if let Some(cfg_dir) = cfg_dir.to_str() {
                groups.set_cfgdir(cfg_dir)?;
            }
        }

        Resolver::from_dynamic_config(groups)
    }

    /// Create a new resolver from a dynamic group configuration
    ///
    /// `set_cfgdir` must already have been called on the dynamic group configuration
    fn from_dynamic_config(groups: DynamicGroupConfig) -> Result<Self, ConfigurationError> {
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
                    let dynamic_groups = DynamicGroupConfig::from_reader(BufReader::new(file))?;
                    resolver.add_sources(dynamic_groups);
                }
            }
        }

        resolver.add_sources(groups);

        Ok(resolver)
    }

    /// Set the global resolver to use for parsing NodeSet using the FromStr trait
    pub fn set_global(resolver: Resolver) {
        *GLOBAL_RESOLVER.write().unwrap() = Some(Arc::new(resolver));
    }

    /// Get the global resolver
    pub fn get_global() -> Arc<Resolver> {
        GLOBAL_RESOLVER
            .read()
            .unwrap()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Arc::new(Resolver::default()))
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
    ) -> Vec<(String, NodeSet)> {
        self.sources
            .iter()
            .map(|(source, groups)| {
                (
                    source.clone(),
                    Parser::default().parse(&groups.list()).unwrap_or_default(),
                )
            })
            .collect()
    }

    /// Return the default group source
    pub fn default_source(&self) -> &str {
        &self.default_source
    }

    fn add_sources(
        &mut self,
        sources: impl IntoIterator<Item = (String, impl GroupSource + 'static)>,
    ) {
        sources.into_iter().for_each(|(name, source)| {
            self.sources.insert(name, Box::new(source));
        });
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
    }

    #[test]
    fn test_parse_dynamic_config() {
        use tempdir::TempDir;

        let config = include_str!("tests/groups.conf");
        let mut dynamic = DynamicGroupConfig::from_reader(config.as_bytes()).unwrap();

        let tmp_dir = TempDir::new("tests").unwrap();

        std::fs::create_dir(tmp_dir.path().join("groups.d")).unwrap();

        std::fs::write(
            tmp_dir.path().join("groups.d").join("local.cfg"),
            include_str!("tests/local.cfg"),
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
            "compute,gpu,all,adm,io,mds,oss".parse::<NodeSet>().unwrap()
        );
    }
}
