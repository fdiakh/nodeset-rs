
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nodeset.h"

#ifndef DEBUG
#define DEBUG false
#endif

void run_test(bool(test)(), const char *name, bool *passed) {
    if (!test()) {
        *passed = false;
        fprintf(stderr, "FAILED: %s\n", name);
    } else {
        fprintf(stderr, "PASSED: %s\n", name);
    }
}

bool test_ns_list() {
    char *error = NULL;
    char **nodes = NULL;
    size_t len = 0;

    if (ns_list("a,b,c", &len, &nodes, &error)) {
        fprintf(stderr, "Error: %s\n", error);
        ns_free_error(error);
        return false;
    }

    if (len != 3) {
        fprintf(stderr, "ns_list returned %ld nodes instead of 3\n", len);
        return false;
    }

    if (strcmp(nodes[0], "a") != 0) {
        fprintf(stderr, "ns_list returned node %s instead of a\n", nodes[0]);
        return false;
    }

    if (strcmp(nodes[1], "b") != 0) {
        fprintf(stderr, "ns_list returned node %s instead of b\n", nodes[1]);
        return false;
    }

    if (strcmp(nodes[2], "c") != 0) {
        fprintf(stderr, "ns_list returned node %s instead of c\n", nodes[2]);
        return false;
    }

    ns_free_node_list(nodes, len);

    return true;
}

bool test_ns_parse() {
    char *error = NULL;
    NodeSet *nodeset = NULL;

    if ((nodeset = ns_parse("n[1-5]", &error)) == NULL) {
        fprintf(stderr, "Error: %s\n", error);
        ns_free_error(error);
        return false;
    }

    ns_free_nodeset(nodeset);

    if ((nodeset = ns_parse("n[1-5", &error)) != NULL) {
        fprintf(stderr, "Error: ns_parse should have failed\n");
    }

    if (error == NULL) {
        fprintf(stderr, "Error: ns_parse should have set an error\n");
        return false;
    }

    ns_free_error(error);

    return true;
}

bool test_ns_count() {
    NodeSet *nodeset = ns_parse("n[1-5]", NULL);
    size_t len = ns_count(nodeset);
    ns_free_nodeset(nodeset);

    if (len != 5) {
        fprintf(stderr, "Error: ns_count returned unexpected length\n");
        return false;
    }

    return true;
}

bool test_ns_fold() {
    NodeSet *nodeset = ns_parse("n[1-5]", NULL);
    char *folded = NULL;
    if ((folded = ns_fold(nodeset, NULL)) == NULL) {
        return false;
    }

    if (strcmp(folded, "n[1-5]") != 0) {
        fprintf(stderr, "Error: ns_fold returned unexpected string\n");
        return false;
    }

    ns_free_nodeset(nodeset);
    ns_free_node(folded);

    return true;
}

bool test_ns_intersection() {
    NodeSet *nodeset1 = ns_parse("n[1-5]", NULL);
    NodeSet *nodeset2 = ns_parse("n[2-6]", NULL);
    NodeSet *intersection = ns_intersection(nodeset1, nodeset2);

    if (ns_count(intersection) != 4) {
        fprintf(stderr, "Error: ns_intersection returned unexpected length\n");
        return false;
    }

    ns_free_nodeset(nodeset1);
    ns_free_nodeset(nodeset2);
    ns_free_nodeset(intersection);

    return true;
}

bool test_ns_union() {
    NodeSet *nodeset1 = ns_parse("n[1-5]", NULL);
    NodeSet *nodeset2 = ns_parse("n[2-6]", NULL);
    NodeSet *union_ = ns_union(nodeset1, nodeset2);

    if (ns_count(union_) != 6) {
        fprintf(stderr, "Error: ns_union returned unexpected length\n");
        return false;
    }

    ns_free_nodeset(nodeset1);
    ns_free_nodeset(nodeset2);
    ns_free_nodeset(union_);

    return true;
}

bool test_ns_symmetric_difference() {
    NodeSet *nodeset1 = ns_parse("n[1-5]", NULL);
    NodeSet *nodeset2 = ns_parse("n[2-6]", NULL);
    NodeSet *symdiff = ns_symmetric_difference(nodeset1, nodeset2);

    if (ns_count(symdiff) != 2) {
        fprintf(stderr,
                "Error: ns_symmetric_difference returned unexpected length\n");
        return false;
    }

    ns_free_nodeset(nodeset1);
    ns_free_nodeset(nodeset2);
    ns_free_nodeset(symdiff);

    return true;
}

bool test_ns_difference() {
    NodeSet *nodeset1 = ns_parse("n[1-5]", NULL);
    NodeSet *nodeset2 = ns_parse("n[2-6]", NULL);
    NodeSet *diff = ns_difference(nodeset1, nodeset2);

    if (ns_count(diff) != 1) {
        fprintf(stderr, "Error: ns_difference returned unexpected length\n");
        return false;
    }

    ns_free_nodeset(nodeset1);
    ns_free_nodeset(nodeset2);
    ns_free_nodeset(diff);

    return true;
}

int main(int argc, char **argv) {
    bool passed = true;

    run_test(test_ns_list, "ns_list", &passed);
    run_test(test_ns_parse, "ns_parse", &passed);
    run_test(test_ns_count, "ns_count", &passed);
    run_test(test_ns_fold, "ns_fold", &passed);
    run_test(test_ns_intersection, "ns_intersection", &passed);
    run_test(test_ns_union, "ns_union", &passed);
    run_test(test_ns_symmetric_difference, "ns_symmetric_difference", &passed);
    run_test(test_ns_difference, "ns_difference", &passed);

    if (!passed) {
        exit(1);
    }
    return 0;
}
