#include <nodeset.h>
#include <stdio.h>

/*
 * This examples shows how to list nodes in a nodeset using the C API
 */
int main(int argc, char **argv) {
    char *error;
    char **nodes;
    size_t len;
    int i;

    if (argc != 2) {
        fprintf(stderr, "Usage: ns_list '<nodeset>'\n\n");
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "\tns_list 'n[1-10]!n[2-9]'\n");
        return 1;
    }

    /* Initialize the group resolver with default settings */
    if (init_default_resolver(&error) != 0) {
        fprintf(stderr, "error: %s\n", error);
        ns_free_error(error);
        return 1;
    }

    /* Compute a list of nodes and store it an array */
    if (ns_list(argv[1], &len, &nodes, &error) != 0) {
        fprintf(stderr, "error: %s\n", error);
        ns_free_error(error);
        return 1;
    }

    /* Display each node */
    for (i = 0; i < len; ++i) {
        printf("%s\n", nodes[i]);
    }

    /* Free the nodes array */
    ns_free_node_list(nodes, len);

    return 0;
}
