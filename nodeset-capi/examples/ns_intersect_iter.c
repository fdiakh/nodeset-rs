#include <nodeset.h>
#include <stdio.h>

/*
 * This examples shows how to iterate over the intersection of two nodesets
 * using the C API
 */
int main(int argc, char **argv) {
    char *error;
    char *node;
    int rc;
    NodeSetIter *iter;
    NodeSet *nodeset1, *nodeset2, *intersection;

    if (argc != 3) {
        fprintf(stderr,
                "usage: ns_intersect_iter '<nodeset1>' '<nodeset2>'\n\n");
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "\tns_intersect_iter 'n[1-8]' 'n[6-10]'\n");
        return 1;
    }

    /* Initialize the group resolver with default settings */
    if (init_default_resolver(&error) != 0) {
        fprintf(stderr, "error: %s\n", error);
        ns_free_error(error);
        return 1;
    }

    /* Parse the nodesets */
    if ((nodeset1 = ns_parse(argv[1], &error)) == NULL) {
        fprintf(stderr, "error: %s\n", error);
        ns_free_error(error);
        return 1;
    }

    if ((nodeset2 = ns_parse(argv[2], &error)) == NULL) {
        fprintf(stderr, "error: %s\n", error);
        ns_free_error(error);
        ns_free_nodeset(nodeset1);
        return 1;
    }

    /* Compute the intersection between both nodesets */
    intersection = ns_intersection(nodeset1, nodeset2);
    ns_free_nodeset(nodeset2);
    ns_free_nodeset(nodeset1);

    /* Create an iterator over the nodes in the nodeset */
    iter = ns_iter(intersection);

    /* Iterate over the nodes */
    while ((node = ns_iter_next(iter, &error)) != NULL) {
        printf("%s\n", node);
        ns_free_node(node);
    }

    /* Check whether the iterator ended due to an error */
    if (ns_iter_status(iter) != 0) {
        fprintf(stderr, "error: %s\n", error);
        ns_free_error(error);
        rc = 1;
    } else {
        rc = 0;
    }

    /* Free the iterator and nodeset */
    ns_free_iter(iter);
    ns_free_nodeset(intersection);

    return rc;
}
