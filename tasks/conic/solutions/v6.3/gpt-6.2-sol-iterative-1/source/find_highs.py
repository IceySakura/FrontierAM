"""Generate an optional HiGHS ABI map for the sealed SciPy wheel."""
import glob
import hashlib
import json
import os
import subprocess
import sysconfig

SYMBOLS = {
    "constructor": "_ZN5HighsC1Ev",
    "destructor": "_ZN5HighsD1Ev",
    "set_b": "_ZN5Highs14setOptionValueERKSsb",
    "set_i": "_ZN5Highs14setOptionValueERKSsi",
    "set_d": "_ZN5Highs14setOptionValueERKSsd",
    "set_s": "_ZN5Highs14setOptionValueERKSsPKc",
    "pass_lp": "_ZN5Highs9passModelEiiiiidPKdS1_S1_S1_S1_PKiS3_S1_S3_",
    "pass_p": "_ZN5Highs11passHessianEiiiPKiS1_PKd",
    "run": "_ZN5Highs3runEv",
    "status": "_Z20highs_getModelStatusP5Highs",
    "solution": "_ZNK5Highs11getSolutionEv",
    "primal_ray": "_ZN5Highs12getPrimalRayERbPd",
    "dual_ray": "_ZN5Highs10getDualRayERbPd",
    "add_rows": "_ZN5Highs7addRowsEiPKdS1_iPKiS3_S1_",
    "change_bounds": "_ZN5Highs16changeColsBoundsEiiPKdS1_",
}
EXPECTED = "47a281d9ef019f205fa24b97049c61aff6daca24b92e3c278b3642ce53a652fa"


def main():
    paths = glob.glob(os.path.join(sysconfig.get_path("platlib"),
                                   "scipy", "optimize", "_highspy", "_core.*.so"))
    if len(paths) != 1:
        return
    path = paths[0]
    with open(path, "rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != EXPECTED:
            return
    try:
        entries = subprocess.check_output(["nm", path], text=True)
    except (OSError, subprocess.CalledProcessError):
        return
    offsets = {}
    reverse = {v: k for k, v in SYMBOLS.items()}
    for line in entries.splitlines():
        rec = line.split()
        if len(rec) == 3 and rec[2] in reverse:
            offsets[reverse[rec[2]]] = int(rec[0], 16)
    if not all(offsets.get(k) for k in SYMBOLS):
        return
    print("#define SEALED_HIGHS 1")
    print("constexpr const char* wheel=" + json.dumps(path) + ";")
    print("constexpr const char* libpython=" + json.dumps(os.path.join(
        sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY"))) + ";")
    for k, v in offsets.items():
        print(f"constexpr size_t ofs_{k}={v};")


if __name__ == "__main__":
    main()
