use super::parsers::Parser;
use super::NodeSet;
use crate::idrange::IdRange;
use crate::NodeSetParseError;
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

#[derive(Deserialize, Debug)]
struct StaticGroupSource {
    #[serde(flatten)]
    groups: HashMap<String, SingleOrVec>,
}

#[derive(Deserialize, Debug)]
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

#[derive(Deserialize, Debug, Default)]
struct DynamicGroupConfig {
    #[serde(rename = "Main")]
    config: Option<ResolverConfig>,
    #[serde(flatten)]
    groups: HashMap<String, DynamicGroupSource>,
}

#[derive(Deserialize, Debug)]
struct DynamicGroupSource {
    map: String,
    all: Option<String>,
    list: Option<String>,
}

trait GroupSource: Debug + Send + Sync {
    fn map(&self, group: &str) -> Result<Option<String>, NodeSetParseError>;
    fn list(&self) -> Vec<String>;
}

impl DynamicGroupConfig {
    fn from_reader(reader: impl std::io::Read) -> Result<Self, NodeSetParseError> {
        let config: Self = serde_ini::from_read(reader)?;
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

    fn set_cfgdir(&mut self, cfgdir: &str) -> Result<(), NodeSetParseError> {
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
            group.map = env_with_context_no_errors(&group.map, context).to_string();
            group.all = group
                .all
                .as_ref()
                .map(|s| env_with_context_no_errors(s, context).to_string());
            group.list = group
                .list
                .as_ref()
                .map(|s| env_with_context_no_errors(s, context).to_string());
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
        Ok(Some(String::from_utf8_lossy(&output.stdout).to_string()))
    }

    fn list(&self) -> Vec<String> {
        unimplemented!()
    }
}

impl StaticGroupConfig {
    fn from_reader(reader: impl std::io::Read) -> Result<Self, NodeSetParseError> {
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

    fn list(&self) -> Vec<String> {
        self.groups.keys().cloned().collect()
    }
}

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
    pub fn from_config() -> Result<Self, NodeSetParseError> {
        let mut resolver = Resolver::default();
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

        resolver.default_source = groups
            .config
            .as_ref()
            .and_then(|c| c.default.clone())
            .unwrap_or_else(|| "default".to_string());

        for autodir in groups.autodirs() {
            for path in find_files_with_ext(Path::new(&autodir), "yaml") {
                if let Some(file) = open_config_path(&path) {
                    let static_groups = StaticGroupConfig::from_reader(BufReader::new(file))?;
                    resolver.add_sources(static_groups);
                }
            }
        }
        for confdir in groups.confdirs() {
            for path in find_files_with_ext(Path::new(&confdir), "cfg") {
                if let Some(file) = open_config_path(&path) {
                    let dynamic_groups = DynamicGroupConfig::from_reader(BufReader::new(file))?;
                    resolver.add_sources(dynamic_groups);
                }
            }
        }

        resolver.add_sources(groups);

        Ok(resolver)
    }

    pub fn set_global(resolver: Resolver) {
        *GLOBAL_RESOLVER.write().unwrap() = Some(Arc::new(resolver));
    }

    pub fn get_global() -> Arc<Resolver> {
        GLOBAL_RESOLVER
            .read()
            .unwrap()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Arc::new(Resolver::default()))
    }

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
    fn test_parse_static_config() {
        let config = include_str!("tests/cluster.yaml");
        let mut resolver = Resolver::default();
        resolver.add_sources(StaticGroupConfig::from_reader(config.as_bytes()).unwrap());
        let parser = Parser::with_resolver(&resolver, Some("roles"));
        assert_eq!(
            parser.parse::<IdRangeList>("@login").unwrap().to_string(),
            "login[1-2]"
        );
    }

    #[test]
    fn test_parse_dynamic_config() {
        let config = include_str!("tests/groups.cfg");
        let _ = DynamicGroupConfig::from_reader(config.as_bytes()).unwrap();
    }
}
