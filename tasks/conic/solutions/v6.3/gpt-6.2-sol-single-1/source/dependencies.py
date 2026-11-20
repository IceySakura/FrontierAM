"""Resolve numerical entry points in the pinned offline SciPy wheel at build time."""
from pathlib import Path
import json
import re
import subprocess
import sysconfig


site = Path(sysconfig.get_path("platlib"))
scipy = site / "scipy"
highs = next((scipy / "optimize" / "_highspy").glob("_core*.so"), None)
superlu = next((scipy / "sparse" / "linalg" / "_dsolve").glob("_superlu*.so"), None)


def symbols(path):
    if path is None:
        return {}
    ret = subprocess.run(["nm", "--defined-only", str(path)], capture_output=True, text=True)
    return {row[2]: int(row[0], 16) for line in ret.stdout.splitlines()
            if len(row := line.split()) == 3}


highs_names = {
    "constructor": "_ZN5HighsC1Ev",
    "destructor": "_ZN5HighsD1Ev",
    "run": "_ZN5Highs3runEv",
    "set_b": "_ZN5Highs14setOptionValueERKSsb",
    "set_i": "_ZN5Highs14setOptionValueERKSsi",
    "set_d": "_ZN5Highs14setOptionValueERKSsd",
    "set_s": "_ZN5Highs14setOptionValueERKSsPKc",
    "pass_lp": "_ZN5Highs9passModelEiiiiidPKdS1_S1_S1_S1_PKiS3_S1_S3_",
    "pass_qp": "_ZN5Highs9passModelEiiiiiiidPKdS1_S1_S1_S1_PKiS3_S1_S3_S3_S1_S3_",
    "get_solution": "_ZNK5Highs11getSolutionEv",
    "get_dual_ray": "_ZN5Highs10getDualRayERbPd",
    "get_primal_ray": "_ZN5Highs12getPrimalRayERbPd",
    "add_rows": "_ZN5Highs7addRowsEiPKdS1_iPKiS3_S1_",
}
hs = symbols(highs)
sl = symbols(superlu)

status_member = 0
if highs is not None:
    clear = subprocess.run(["objdump", "-d", "--disassemble=_ZN5Highs21invalidateModelStatusEv",
                            str(highs)], capture_output=True, text=True).stdout
    match = re.search(r"\$0x0,0x([0-9a-f]+)\(%rdi\)", clear)
    if match:
        status_member = int(match.group(1), 16)

print("#pragma once")
print("namespace installed {")
print("constexpr const char* highs =", json.dumps(str(highs or "")), ";")
print("constexpr const char* superlu =", json.dumps(str(superlu or "")), ";")
for name, mangled in highs_names.items():
    print(f"constexpr unsigned hi_{name} = {hs.get(mangled, 0)};")
print(f"constexpr unsigned hi_status_member = {status_member};")
for name in ("genmmd_", "colamd", "colamd_recommended"):
    print(f"constexpr unsigned sl_{name} = {sl.get(name, 0)};")
print("}")
