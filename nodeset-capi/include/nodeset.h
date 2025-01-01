#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>


/**
 * An iterator over nodes in a nodeset
 */
typedef struct NodeSetIter NodeSetIter;

/**
 * An unordered collection of nodes indexed in one or more dimensions.
 *
 * Two implementations are provided:
 * * `NodeSet<IdRangeList>` which stores node indices in Vecs
 * * `NodeSet<IdRangeTree>` which stores node indices in BTrees
 *
 * By default `IdRangeList` are used as they are faster to build for one shot
 * operations which are the most common, especially when using the CLI.
 * However, if many updates are performed on a large NodeSet `IdRangeTree` may
 * more efficient especially for one-dimensional NodeSets.
 */
typedef struct NodeSet_IdRangeList NodeSet_IdRangeList;

typedef struct NodeSet_IdRangeList NodeSet;

/**
 * Initialize the group resolver with default settings
 *
 * `init_default_resolver()` initializes the group resolver based on
 * clustershell's configuration files.
 *
 * In case of error, `*error` is set to a newly allocated string containing the
 * error message unless NULL was passed.
 *
 * # Safety
 *
 * The caller must free the error string by calling `ns_free_error()` in case
 * of error.
 *
 * # Return value
 *
 * Returns 0 on success, -1 on error.
 *
 */
int init_default_resolver(char **error);

/**
 * Count the number of nodes in a nodeset
 *
 * # Safety
 *
 * `nodeset` must be a valid nodeset returned by this library.
 *
 */
size_t ns_count(NodeSet *nodeset);

/**
 * Compute the difference of two nodesets
 *
 * `ns_difference()` returns a new nodeset containing nodes that are in `nodeset1` but
 * not in `nodeset2`.
 *
 * # Safety
 *
 * `nodeset1` and `nodeset2` must be valid nodesets returned by this library.
 *
 * The resulting nodeset must be freed by calling `ns_free_nodeset()`.
 *
 */
NodeSet *ns_difference(NodeSet *nodeset1, NodeSet *nodeset2);

/**
 * Compute a folded representation of a nodeset
 *
 * `ns_fold()` returns a string containing the folded representation of the
 * nodeset. NULL is returned if an error occured and `*error` is set to a newly
 * allocated string containing the error message unless NULL was passed.
 *
 * # Safety
 *
 * `nodeset` must be a valid nodeset returned by this library.
 *
 */
char *ns_fold(NodeSet *nodeset, char **error);

/**
 * Free an error string returned by any function in this library
 *
 * # Safety
 *
 * `error` must be provided as returned by a function of this library and its
 * content must not have been modified
 *
 */
void ns_free_error(char *error);

/**
 * Free an iterator returned by `ns_iter()`
 *
 * # Safety
 *
 * `iter` must be provided as returned by `ns_iter()` and its content must not
 * have been modified
 *
 */
void ns_free_iter(struct NodeSetIter *iter);

/**
 * Free a node string returned by any function in this library
 *
 * # Safety
 *
 * `node` must be provided as returned by a function of this library and its
 * content must not have been modified
 *
 */
void ns_free_node(char *node);

/**
 * Free a node array returned by `ns_list()`
 *
 * # Safety
 *
 * Both `nodes` and `len` must be provided as returned by `ns_list()` and their
 * content must not have been modified
 *
 */
void ns_free_node_list(char **nodes, size_t len);

/**
 * Free a nodeset returned by `ns_parse()`
 *
 * # Safety
 *
 * `nodeset` must be a valid nodeset returned by this library.
 *
 */
void ns_free_nodeset(NodeSet *nodeset);

/**
 * Compute the intersection of two nodesets
 *
 * `ns_intersection()` returns a new nodeset containing nodes that are in both
 * `nodeset1` and `nodeset2`.
 *
 * # Safety
 *
 * `nodeset1` and `nodeset2` must be valid nodesets returned by this library.
 *
 *  The resulting nodeset must be freed by calling `ns_free_nodeset()`.
 *
 */
NodeSet *ns_intersection(NodeSet *nodeset1, NodeSet *nodeset2);

/**
 * Create an iterator over the nodes in a nodeset.
 *
 * `ns_iter()` returns a newly allocated iterator iterator over the nodes in
 * `nodeset`. nodeset.
 *
 * # Safety
 *
 * The caller must free the iterator by calling `ns_free_iter()`. The nodeset
 * must not be freed until the iterator is freed.
 *
 */
struct NodeSetIter *ns_iter(NodeSet *nodeset);

/**
 * Get the next node from an iterator
 *
 * `ns_iter_next()` returns the name of the next node yielded by the iterator
 * `iter`. It returns NULL if it cannot provide more nodes, due to an error or
 * to the iterator being depleted. In case of error, `*error` is set to a newly
 * allocated string containing the error message unless NULL was passed.
 *
 * `ns_iter_status()` should be called after `ns_iter_next()` returns NULL to check
 * whether the iterator ended due to an error.
 *
 * # Safety
 *
 * `iter` must be a valid iterator returned by `ns_iter()`.
 *
 * In case of error, the caller must free the error string by calling
 * `ns_free_error()`.
 *
 *
 */
char *ns_iter_next(struct NodeSetIter *iter, char **error);

/**
 * Returns whether the iterator has encountered an error
 *
 * `ns_iter_status()` returns -1 if the iterator has failed, 0 otherwise.
 *
 * This should be called after `ns_iter_next()` returns NULL to check whether the
 * iterator ended due to an error.
 *
 * # Safety
 *
 * `iter` must be a valid iterator returned by `ns_iter()`.
 *
 */
int ns_iter_status(struct NodeSetIter *iter);

/**
 * Expand a nodeset into a list of nodes
 *
 * `ns_list()` parses the nodeset expression passed in `nodeset` and sets
 * `*nodes` to a newly allocated array of node names on success. `*len` is set
 *  to the number of nodes. The nodeset expression may contain operators and
 * groups. In case of error, `*error` is set to a newly allocated string
 * containing the error message unless NULL was passed.
 *
 * # Safety
 *
 * The caller must free the nodes array by calling `ns_free_node_list()` in
 * case of success.
 *
 * The caller must free the error string by calling `ns_free_error()` in case
 * of error.
 *
 * # Return value
 *
 * Returns 0 on success, 1 on error.
 *
 */
int ns_list(const char *nodeset, size_t *len, char ***nodes, char **error);

/**
 * Parse a nodeset
 *
 * `ns_parse()` parses the nodeset expression passed in `nodeset` and returns a
 *  nodeset. The expression may contain operators and groups. In case of error,
 * NULL is returned and `*error` is set to a newly allocated string containing
 * the error message unless NULL was passed.
 *
 * # Safety
 *
 * The caller must free the nodeset by calling `ns_free_nodeset()` in case of
 * success.
 *
 * The caller must free the error string by calling `ns_free_error()` in case
 * of error.
 */
NodeSet *ns_parse(const char *nodeset, char **error);

/**
 * Compute the symmetric difference of two nodesets
 *
 * `ns_symmetric_difference()` returns a new nodeset containing nodes that are in either
 * `nodeset1` or `nodeset2` but not in both.
 *
 * # Safety
 *
 * `nodeset1` and `nodeset2` must be valid nodesets returned by this library.
 *
 * The resulting nodeset must be freed by calling `ns_free_nodeset()`.
 *
 */
NodeSet *ns_symmetric_difference(NodeSet *nodeset1, NodeSet *nodeset2);

/**
 * Compute the union of two nodesets
 *
 * `ns_union()` returns a new nodeset containing nodes that are in either
 * `nodeset1` or `nodeset2`.
 *
 * # Safety
 *
 * `nodeset1` and `nodeset2` must be valid nodesets returned by this library.
 *
 *  The resulting nodeset must be freed by calling `ns_free_nodeset()`.
 *
 */
NodeSet *ns_union(NodeSet *nodeset1, NodeSet *nodeset2);
