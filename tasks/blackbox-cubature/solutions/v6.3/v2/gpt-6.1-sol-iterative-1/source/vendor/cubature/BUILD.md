# Fixed Source And Build

This directory is a verbatim source snapshot of `stevengj/cubature` at commit
`4ba70a7d595a00cfb15b02763013e0088013adb7` (v1.0.3 lineage). The upstream
source is GPL v2 or later; `COPYING.md`, `README.md` and `CITATION.bib` are
retained. It is used only by the maintenance-side baseline runner, not exposed
to the Agent as a function implementation.

This submission vendors the p-adaptive implementation and its required
headers. The numeric tables are verbatim prefixes through level 12 of the
provided level-19 tables. The 2048-point budget cannot reach a larger rule.

Build from the submission directory on the same CPU image:

```sh
cc -O2 -fPIC -shared vendor/cubature/pcubature.c -o libpcubature.so -lm
```

The Python callback rejects a batch before querying if it would exceed the
remaining budget. A rejected expansion leaves the library's previous complete
estimate intact.
