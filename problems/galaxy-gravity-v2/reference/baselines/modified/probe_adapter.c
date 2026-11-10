// SPDX-License-Identifier: GPL-3.0-or-later
#include "clib/stdinc.h"
#include "clib/mathfns.h"
#include "clib/vectmath.h"
#include "clib/getparam.h"
#include "treedefs.h"
#include <limits.h>
#include <sys/stat.h>

int main(int argc, char **argv)
{
    if (argc < 5 || argc > 7)
        error("usage: %s INPUT OUTPUT THETA EPS [NAVG [SEED]]\n", argv[0]);
    const double theta = atof(argv[3]), epsilon = atof(argv[4]);
    const int ntrees = argc >= 6 ? atoi(argv[5]) : 1;
    const unsigned long seed = argc >= 7 ? strtoul(argv[6], NULL, 10) : 123;
    if (!(theta > 0 && theta <= 1 && epsilon > 0 && ntrees > 0))
        error("invalid probe controls\n");
    struct stat st;
    if (stat(argv[1], &st) || st.st_size < 8 * sizeof(double) || st.st_size % (4 * sizeof(double)) ||
        st.st_size / (4 * sizeof(double)) > INT_MAX)
        error("invalid raw input size\n");
    const int n = st.st_size / (4 * sizeof(double));
    bodyptr particles = calloc(n, sizeof(body));
    double (*out)[3] = calloc(n, sizeof(*out));
    FILE *input = fopen(argv[1],"rb");
    if (!particles || !out || !input) error("could not load input\n");
    for (int i=0; i<n; i++) {
        double record[4];
        if (fread(record, sizeof(record), 1, input) != 1)
            error("short input at row %d\n", i);
        Type(&particles[i]) = BodyType;
        Mass(&particles[i]) = record[0];
        memcpy(Pos(&particles[i]), record + 1, 3 * sizeof(double));
    }
    fclose(input);
    setenv("ZENO_MSG_OPTION", "warn", 1);
    string init_args[] = {argv[0], NULL}, defs[] = {NULL};
    initparam(init_args,defs);
    init_random(seed);
    for (int t=0; t<ntrees; t++) {
        for (int i=0; i<n; i++) Active(&particles[i]) = 1;
        // Upstream treebuild transforms ONLY partition coordinates; treegrav
        // returns accelerations in physical coordinates for every tree.
        const bool randomize = ntrees > 1;
        treeptr tree = treebuild(particles, n, epsilon, theta, TRUE,
                                 randomize, randomize, randomize ? 1.0 : 0.0, "");
#ifdef MODTREECODE
        treegrav(tree, n > 64 ? 64 : n - 1);
#else
        treegrav(tree);
#endif
        for (int i=0; i<n; i++)
            for (int k=0; k<3; k++) out[i][k] += Acc(&particles[i])[k];
        treedestroy(tree);
    }
    for (int i=0; i<n; i++)
        for (int k=0; k<3; k++) out[i][k] /= ntrees;
    FILE *output = fopen(argv[2], "wb");
    if (!output || fwrite(out, sizeof(*out), n, output) != (size_t)n || fclose(output))
        error("could not write output\n");
    free(particles);
    free(out);
    return 0;
}
