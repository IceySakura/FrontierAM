// SPDX-License-Identifier: GPL-3.0-or-later
// Pin the probe's one-tree invocation without starting another process.
#define main probe_main
#include "probe_adapter.c"
#undef main

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    char *args[] = {argv[0], argv[1], argv[2], "0.8", "0.01", NULL};
    return probe_main(5, args);
}
