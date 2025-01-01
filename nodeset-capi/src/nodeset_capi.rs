use anyhow::Result;
use nodeset::{IdRangeList, NodeSetIter as RustNodeSetIter, Resolver};
use std::ffi::{c_int, CStr, CString};

use std::os::raw::c_char;

const DEFAULT_ERROR: &str = "Unknown error";

type NodeSet = nodeset::NodeSet<IdRangeList>;

#[no_mangle]
/// Initialize the group resolver with default settings
///
/// `init_default_resolver()` initializes the group resolver based on
/// clustershell's configuration files.
///
/// In case of error, `*error` is set to a newly allocated string containing the
/// error message unless NULL was passed.
///
/// # Safety
///
/// The caller must free the error string by calling `ns_free_error()` in case
/// of error.
///
/// # Return value
///
/// Returns 0 on success, -1 on error.
///
pub unsafe extern "C" fn init_default_resolver(error: *mut *mut c_char) -> c_int {
    let res = (|| -> Result<()> {
        let _ = Resolver::set_global(Resolver::from_config()?);
        Ok(())
    })();

    match res {
        Ok(_) => 0,
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = CString::new(format!("{}", e))
                        .unwrap_or_else(|_| CString::new(DEFAULT_ERROR).unwrap())
                        .into_raw();
                }
            }
            -1
        }
    }
}

#[no_mangle]
/// Parse a nodeset
///
/// `ns_parse()` parses the nodeset expression passed in `nodeset` and returns a
///  nodeset. The expression may contain operators and groups. In case of error,
/// NULL is returned and `*error` is set to a newly allocated string containing
/// the error message unless NULL was passed.
///
/// # Safety
///
/// The caller must free the nodeset by calling `ns_free_nodeset()` in case of
/// success.
///
/// The caller must free the error string by calling `ns_free_error()` in case
/// of error.
///
pub unsafe extern "C" fn ns_parse(nodeset: *const c_char, error: *mut *mut c_char) -> *mut NodeSet {
    let res = (|| -> Result<*mut NodeSet> {
        let nodeset = unsafe { CStr::from_ptr(nodeset) }.to_str()?;
        let nodeset: NodeSet = nodeset.parse()?;
        Ok(Box::into_raw(Box::new(nodeset)))
    })();

    match res {
        Ok(res) => res,
        Err(e) => {
            ffi_error(e, error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
/// Expand a nodeset into a list of nodes
///
/// `ns_list()` parses the nodeset expression passed in `nodeset` and sets
/// `*nodes` to a newly allocated array of node names on success. `*len` is set
///  to the number of nodes. The nodeset expression may contain operators and
/// groups. In case of error, `*error` is set to a newly allocated string
/// containing the error message unless NULL was passed.
///
/// # Safety
///
/// The caller must free the nodes array by calling `ns_free_node_list()` in
/// case of success.
///
/// The caller must free the error string by calling `ns_free_error()` in case
/// of error.
///
/// # Return value
///
/// Returns 0 on success, 1 on error.
///
pub unsafe extern "C" fn ns_list(
    nodeset: *const c_char,
    len: *mut usize,
    nodes: *mut *mut *mut c_char,
    error: *mut *mut c_char,
) -> c_int {
    let mut c_nodes = vec![];

    let res = (|| -> Result<()> {
        let nodeset = unsafe { CStr::from_ptr(nodeset) }.to_str()?;
        let nodeset: NodeSet = nodeset.parse()?;

        for node in nodeset.iter() {
            let c_node = CString::new(node)?;
            c_nodes.push(c_node.into_raw());
        }

        Ok(())
    })();

    match res {
        Ok(_) => {
            unsafe { *len = c_nodes.len() }
            let c_nodes = c_nodes.into_boxed_slice();
            *nodes = Box::into_raw(c_nodes) as *mut *mut c_char;
            0
        }
        Err(e) => {
            for node in c_nodes {
                unsafe {
                    let _ = CString::from_raw(node);
                };
            }
            ffi_error(e, error);

            1
        }
    }
}

#[no_mangle]
/// Compute a folded representation of a nodeset
///
/// `ns_fold()` returns a string containing the folded representation of the
/// nodeset. NULL is returned if an error occured and `*error` is set to a newly
/// allocated string containing the error message unless NULL was passed.
///
/// # Safety
///
/// `nodeset` must be a valid nodeset returned by this library.
///
pub unsafe extern "C" fn ns_fold(nodeset: *mut NodeSet, error: *mut *mut c_char) -> *mut c_char {
    let res = (|| -> Result<CString> {
        let rust_nodeset = unsafe { Box::from_raw(nodeset) };
        let nodeset = rust_nodeset.to_string();
        let _ = Box::into_raw(rust_nodeset);
        Ok(CString::new(nodeset)?)
    })();

    match res {
        Ok(res) => res.into_raw(),
        Err(e) => {
            ffi_error(e, error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
/// Count the number of nodes in a nodeset
///
/// # Safety
///
/// `nodeset` must be a valid nodeset returned by this library.
///
pub unsafe extern "C" fn ns_count(nodeset: *mut NodeSet) -> usize {
    let rust_nodeset = unsafe { Box::from_raw(nodeset) };
    let res = rust_nodeset.len();
    let _ = Box::into_raw(rust_nodeset);

    res
}

#[no_mangle]
/// Compute the intersection of two nodesets
///
/// `ns_intersection()` returns a new nodeset containing nodes that are in both
/// `nodeset1` and `nodeset2`.
///
/// # Safety
///
/// `nodeset1` and `nodeset2` must be valid nodesets returned by this library.
///
///  The resulting nodeset must be freed by calling `ns_free_nodeset()`.
///
pub unsafe extern "C" fn ns_intersection(
    nodeset1: *mut NodeSet,
    nodeset2: *mut NodeSet,
) -> *mut NodeSet {
    let rust_nodeset1 = unsafe { Box::from_raw(nodeset1) };
    let rust_nodeset2 = unsafe { Box::from_raw(nodeset2) };
    let res = rust_nodeset1.intersection(&rust_nodeset2);
    let _ = Box::into_raw(rust_nodeset1);
    let _ = Box::into_raw(rust_nodeset2);
    Box::into_raw(Box::new(res))
}

#[no_mangle]
/// Compute the union of two nodesets
///
/// `ns_union()` returns a new nodeset containing nodes that are in either
/// `nodeset1` or `nodeset2`.
///
/// # Safety
///
/// `nodeset1` and `nodeset2` must be valid nodesets returned by this library.
///
///  The resulting nodeset must be freed by calling `ns_free_nodeset()`.
///
pub unsafe extern "C" fn ns_union(nodeset1: *mut NodeSet, nodeset2: *mut NodeSet) -> *mut NodeSet {
    let rust_nodeset1 = unsafe { Box::from_raw(nodeset1) };
    let rust_nodeset2 = unsafe { Box::from_raw(nodeset2) };
    let res = rust_nodeset1.union(&rust_nodeset2);
    let _ = Box::into_raw(rust_nodeset1);
    let _ = Box::into_raw(rust_nodeset2);
    Box::into_raw(Box::new(res))
}

#[no_mangle]
/// Compute the symmetric difference of two nodesets
///
/// `ns_symmetric_difference()` returns a new nodeset containing nodes that are in either
/// `nodeset1` or `nodeset2` but not in both.
///
/// # Safety
///
/// `nodeset1` and `nodeset2` must be valid nodesets returned by this library.
///
/// The resulting nodeset must be freed by calling `ns_free_nodeset()`.
///
pub unsafe extern "C" fn ns_symmetric_difference(
    nodeset1: *mut NodeSet,
    nodeset2: *mut NodeSet,
) -> *mut NodeSet {
    let rust_nodeset1 = unsafe { Box::from_raw(nodeset1) };
    let rust_nodeset2 = unsafe { Box::from_raw(nodeset2) };
    let res = rust_nodeset1.symmetric_difference(&rust_nodeset2);
    let _ = Box::into_raw(rust_nodeset1);
    let _ = Box::into_raw(rust_nodeset2);
    Box::into_raw(Box::new(res))
}

#[no_mangle]
/// Compute the difference of two nodesets
///
/// `ns_difference()` returns a new nodeset containing nodes that are in `nodeset1` but
/// not in `nodeset2`.
///
/// # Safety
///
/// `nodeset1` and `nodeset2` must be valid nodesets returned by this library.
///
/// The resulting nodeset must be freed by calling `ns_free_nodeset()`.
///
pub unsafe extern "C" fn ns_difference(
    nodeset1: *mut NodeSet,
    nodeset2: *mut NodeSet,
) -> *mut NodeSet {
    let rust_nodeset1 = unsafe { Box::from_raw(nodeset1) };
    let rust_nodeset2 = unsafe { Box::from_raw(nodeset2) };
    let res = rust_nodeset1.difference(&rust_nodeset2);
    let _ = Box::into_raw(rust_nodeset1);
    let _ = Box::into_raw(rust_nodeset2);
    Box::into_raw(Box::new(res))
}

/// An iterator over nodes in a nodeset
pub struct NodeSetIter<'a> {
    iter: *mut RustNodeSetIter<'a, IdRangeList>,
    has_error: bool,
}

impl<'a> NodeSetIter<'a> {
    unsafe fn new(nodeset: &NodeSet) -> Self {
        // The caller is responsible for not freeing the nodeset before freeing
        // the iterator
        let iter = unsafe {
            std::mem::transmute::<
                *mut RustNodeSetIter<IdRangeList>,
                *mut RustNodeSetIter<'a, IdRangeList>,
            >(Box::into_raw(Box::new(nodeset.iter())))
        };
        Self {
            iter,
            has_error: false,
        }
    }
}

#[no_mangle]
/// Create an iterator over the nodes in a nodeset.
///
/// `ns_iter()` returns a newly allocated iterator iterator over the nodes in
/// `nodeset`. nodeset.
///
/// # Safety
///
/// The caller must free the iterator by calling `ns_free_iter()`. The nodeset
/// must not be freed until the iterator is freed.
///
pub unsafe extern "C" fn ns_iter(nodeset: *mut NodeSet) -> *mut NodeSetIter<'static> {
    let nodeset = unsafe { Box::from_raw(nodeset) };
    let new_iter = NodeSetIter::new(&nodeset);
    let _ = Box::into_raw(nodeset);

    Box::into_raw(Box::new(new_iter))
}

#[no_mangle]
/// Get the next node from an iterator
///
/// `ns_iter_next()` returns the name of the next node yielded by the iterator
/// `iter`. It returns NULL if it cannot provide more nodes, due to an error or
/// to the iterator being depleted. In case of error, `*error` is set to a newly
/// allocated string containing the error message unless NULL was passed.
///
/// `ns_iter_status()` should be called after `ns_iter_next()` returns NULL to check
/// whether the iterator ended due to an error.
///
/// # Safety
///
/// `iter` must be a valid iterator returned by `ns_iter()`.
///
/// In case of error, the caller must free the error string by calling
/// `ns_free_error()`.
///
///
pub unsafe extern "C" fn ns_iter_next(
    iter: *mut NodeSetIter,
    error: *mut *mut c_char,
) -> *mut c_char {
    let res = (|| -> Result<Option<*mut c_char>> {
        let mut c_iter = unsafe { Box::from_raw(iter) };
        let mut rust_iter = unsafe { Box::from_raw(c_iter.iter) };

        let res = rust_iter
            .next()
            .map(|node| CString::new(node).map(|s| s.into_raw()))
            .transpose();

        if res.is_err() {
            c_iter.has_error = true;
        }

        // Forget objects owned by the caller
        let _ = Box::into_raw(rust_iter);
        let _ = Box::into_raw(c_iter);
        Ok(res?)
    })();

    match res {
        Ok(res) => res.unwrap_or(std::ptr::null_mut()),
        Err(e) => {
            ffi_error(e, error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]

/// Returns whether the iterator has encountered an error
///
/// `ns_iter_status()` returns -1 if the iterator has failed, 0 otherwise.
///
/// This should be called after `ns_iter_next()` returns NULL to check whether the
/// iterator ended due to an error.
///
/// # Safety
///
/// `iter` must be a valid iterator returned by `ns_iter()`.
///
pub unsafe extern "C" fn ns_iter_status(iter: *mut NodeSetIter) -> c_int {
    let c_iter = unsafe { Box::from_raw(iter) };
    let res = if c_iter.has_error { -1 } else { 0 };

    let _ = Box::into_raw(c_iter);
    res
}

unsafe fn ffi_error(rust_error: impl std::fmt::Debug, c_error: *mut *mut c_char) {
    if !c_error.is_null() {
        unsafe {
            *c_error = CString::new(format!("{:?}", rust_error))
                .unwrap_or_else(|_| CString::new(DEFAULT_ERROR).unwrap())
                .into_raw();
        }
    }
}

#[no_mangle]
/// Free a nodeset returned by `ns_parse()`
///
/// # Safety
///
/// `nodeset` must be a valid nodeset returned by this library.
///
pub unsafe extern "C" fn ns_free_nodeset(nodeset: *mut NodeSet) {
    unsafe {
        let _ = Box::from_raw(nodeset);
    };
}

#[no_mangle]
/// Free a node array returned by `ns_list()`
///
/// # Safety
///
/// Both `nodes` and `len` must be provided as returned by `ns_list()` and their
/// content must not have been modified
///
pub unsafe extern "C" fn ns_free_node_list(nodes: *mut *mut c_char, len: usize) {
    for node in Vec::from_raw_parts(nodes, len, len) {
        unsafe {
            let _ = CString::from_raw(node);
        };
    }
}

#[no_mangle]
/// Free an error string returned by any function in this library
///
/// # Safety
///
/// `error` must be provided as returned by a function of this library and its
/// content must not have been modified
///
pub unsafe extern "C" fn ns_free_error(error: *mut c_char) {
    unsafe {
        if !error.is_null() {
            let _ = CString::from_raw(error);
        }
    };
}

#[no_mangle]
/// Free a node string returned by any function in this library
///
/// # Safety
///
/// `node` must be provided as returned by a function of this library and its
/// content must not have been modified
///
pub unsafe extern "C" fn ns_free_node(node: *mut c_char) {
    unsafe {
        let _ = CString::from_raw(node);
    };
}

#[no_mangle]
/// Free an iterator returned by `ns_iter()`
///
/// # Safety
///
/// `iter` must be provided as returned by `ns_iter()` and its content must not
/// have been modified
///
pub unsafe extern "C" fn ns_free_iter(iter: *mut NodeSetIter<'static>) {
    unsafe {
        let nsiter = Box::from_raw(iter);
        let _ = Box::from_raw(nsiter.iter);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    #[test]
    fn test_ns_list() {
        let input = CString::new("n[1-5]").expect("CString::new failed");
        let mut len: usize = 0;
        let mut nodes: *mut *mut c_char = std::ptr::null_mut();
        let mut error: *mut c_char = std::ptr::null_mut();

        let r = unsafe { ns_list(input.as_ptr(), &mut len, &mut nodes, &mut error) };

        assert_eq!(r, 0);
        assert_eq!(len, 5);
        assert!(!nodes.is_null());

        unsafe {
            ns_free_node_list(nodes, len);
        }
    }

    #[test]
    fn test_ns_iter() {
        let nodeset = parse_nodeset("n[1-3]");
        let iter = unsafe { ns_iter(nodeset) };

        assert!(!iter.is_null(), "ns_iter() returned null iter");

        let mut nodes = vec![];
        let mut error = std::ptr::null_mut();
        loop {
            let node = unsafe { ns_iter_next(iter, &mut error) };
            if node.is_null() {
                break;
            }

            nodes.push(
                unsafe { CStr::from_ptr(node) }
                    .to_string_lossy()
                    .to_string(),
            );
            unsafe { ns_free_node(node) };
        }

        assert_eq!(
            unsafe { ns_iter_status(iter) },
            0,
            "ns_iter_next() returned an error"
        );

        assert_eq!(
            nodes,
            vec!["n1", "n2", "n3"],
            "ns_iter_next() returned unexpected nodes"
        );

        unsafe {
            ns_free_iter(iter);
            ns_free_nodeset(nodeset);
        };
    }

    #[test]
    fn test_ns_count() {
        let nodeset = parse_nodeset("n[1-5]");

        let len = unsafe { ns_count(nodeset) };
        assert_eq!(len, 5, "ns_count() returned unexpected length");

        unsafe { ns_free_nodeset(nodeset) };
    }

    #[test]
    fn test_ns_intersection() {
        let nodeset1 = parse_nodeset("n[1-5]");
        let nodeset2 = parse_nodeset("n[3-6]");

        let nodeset = unsafe { ns_intersection(nodeset1, nodeset2) };
        assert!(
            !nodeset.is_null(),
            "ns_intersection() returned null nodeset"
        );

        compare_nodeset(nodeset, "n[3-5]");

        unsafe {
            ns_free_nodeset(nodeset1);
            ns_free_nodeset(nodeset2);
            ns_free_nodeset(nodeset);
        };
    }

    #[test]
    fn test_ns_union() {
        let nodeset1 = parse_nodeset("n[1-5]");
        let nodeset2 = parse_nodeset("n[3-6]");

        let nodeset = unsafe { ns_union(nodeset1, nodeset2) };
        assert!(!nodeset.is_null(), "ns_union() returned null nodeset");

        compare_nodeset(nodeset, "n[1-6]");

        unsafe {
            ns_free_nodeset(nodeset1);
            ns_free_nodeset(nodeset2);
            ns_free_nodeset(nodeset);
        };
    }

    #[test]
    fn test_ns_symmetric_difference() {
        let nodeset1 = parse_nodeset("n[1-5]");
        let nodeset2 = parse_nodeset("n[3-6]");

        let nodeset = unsafe { ns_symmetric_difference(nodeset1, nodeset2) };
        assert!(
            !nodeset.is_null(),
            "ns_symmetric_difference() returned null nodeset"
        );

        compare_nodeset(nodeset, "n[1-2,6]");

        unsafe {
            ns_free_nodeset(nodeset1);
            ns_free_nodeset(nodeset2);
            ns_free_nodeset(nodeset);
        };
    }

    #[test]
    fn test_ns_difference() {
        let nodeset1 = parse_nodeset("n[1-5]");
        let nodeset2 = parse_nodeset("n[3-6]");

        let nodeset = unsafe { ns_difference(nodeset1, nodeset2) };
        assert!(!nodeset.is_null(), "ns_difference() returned null nodeset");

        compare_nodeset(nodeset, "n[1-2]");

        unsafe {
            ns_free_nodeset(nodeset1);
            ns_free_nodeset(nodeset2);
            ns_free_nodeset(nodeset);
        };
    }

    #[test]
    fn test_parse_error() {
        let c_input = CString::new("n[1-5").unwrap();
        let mut error: *mut c_char = std::ptr::null_mut();
        let nodeset = unsafe { ns_parse(c_input.as_ptr(), &mut error) };
        assert!(nodeset.is_null());

        assert!(!error.is_null());
        let str_error = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(
            str_error.contains("[1-5"),
            "ns_parse() returned unexpected error"
        );

        unsafe { ns_free_error(error) };
    }

    fn compare_nodeset(nodeset: *mut NodeSet, expected: &str) {
        let res = unsafe { ns_fold(nodeset, std::ptr::null_mut()) };
        assert!(!res.is_null(), "ns_fold() returned null string for nodeset");
        let str_res = unsafe { CStr::from_ptr(res) }.to_str().unwrap();
        assert_eq!(str_res, expected, "ns_fold() returned unexpected string");

        unsafe { ns_free_node(res) };
    }

    fn parse_nodeset(input: &str) -> *mut NodeSet {
        let c_input = CString::new(input).unwrap();
        let mut error: *mut c_char = std::ptr::null_mut();
        let nodeset = unsafe { ns_parse(c_input.as_ptr(), &mut error) };
        assert!(
            !nodeset.is_null(),
            "ns_parse() returned null nodeset for '{}'",
            input
        );
        nodeset
    }
}
