#include <slepcmfn.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEFAULT_TOL_MULTIPLIER 100.0
#define KRYLOV_DIMENSION_ONE_SHOT 60
#define KRYLOV_DIMENSION_REUSE 80

static void die(const char *message)
{
  fprintf(stderr, "stream_slepc: %s\n", message);
  exit(2);
}

static void petsc_check(PetscErrorCode code, const char *operation)
{
  if (code) {
    fprintf(stderr, "stream_slepc: %s failed with PETSc error %d\n", operation, (int)code);
    exit(3);
  }
}

static void read_exact(void *buffer, size_t length)
{
  unsigned char *cursor = buffer;
  while (length) {
    ssize_t count = read(STDIN_FILENO, cursor, length);
    if (count <= 0) die("short input");
    cursor += count;
    length -= (size_t)count;
  }
}

static void write_all(const void *buffer, size_t length)
{
  const unsigned char *cursor = buffer;
  while (length) {
    ssize_t count = write(STDOUT_FILENO, cursor, length);
    if (count <= 0) die("short output");
    cursor += count;
    length -= (size_t)count;
  }
}

static double header_number(const char *json, const char *key)
{
  char quoted[32];
  const char *at;
  snprintf(quoted, sizeof quoted, "\"%s\"", key);
  at = strstr(json, quoted);
  if (!at || !(at = strchr(at + strlen(quoted), ':'))) die("malformed header");
  return strtod(at + 1, NULL);
}

static double parse_multiplier(int argc, char **argv)
{
  char *end = NULL;
  double value;
  if (argc == 1) return DEFAULT_TOL_MULTIPLIER;
  if (argc != 2) die("usage: stream_slepc [tolerance-multiplier]");
  errno = 0;
  value = strtod(argv[1], &end);
  if (errno || end == argv[1] || *end || !isfinite(value) || value <= 0.0)
    die("tolerance multiplier must be finite and positive");
  return value;
}

int main(int argc, char **argv)
{
  double multiplier = parse_multiplier(argc, argv);
  uint32_t json_length;
  char magic[8], json[4097];
  int n, requests;
  uint64_t nnz, *raw_rowptr;
  uint32_t *raw_cols;
  double *raw_values, *b0, *b1, *answer, t, rtol;
  PetscInt *row_nnz, *cols, *rows;
  PetscScalar *matrix_values, *last_column, *input_values;
  Mat A;
  Vec input, output;
  MFN mfn;
  FN fn;

  petsc_check(SlepcInitializeNoArguments(), "SlepcInitialize");
  write_all("READY\n", 6);

  read_exact(magic, sizeof magic);
  if (memcmp(magic, "FAMEXP2\n", sizeof magic)) die("wrong protocol magic");
  read_exact(&json_length, sizeof json_length);
  if (!json_length || json_length > 4096) die("bad header length");
  read_exact(json, json_length);
  json[json_length] = '\0';

  n = (int)header_number(json, "n");
  nnz = (uint64_t)header_number(json, "nnz");
  requests = (int)header_number(json, "requests");
  t = header_number(json, "t");
  rtol = header_number(json, "rtol");
  if (n < 1 || nnz > 1000000 || (requests != 1 && requests != 16)) die("unsupported header");

  raw_rowptr = malloc((size_t)(n + 1) * sizeof *raw_rowptr);
  raw_cols = malloc((size_t)(nnz ? nnz : 1) * sizeof *raw_cols);
  raw_values = malloc((size_t)(nnz ? nnz : 1) * sizeof *raw_values);
  b0 = malloc((size_t)n * sizeof *b0);
  b1 = malloc((size_t)n * sizeof *b1);
  answer = malloc((size_t)n * sizeof *answer);
  row_nnz = malloc((size_t)(n + 1) * sizeof *row_nnz);
  cols = malloc((size_t)(nnz + n) * sizeof *cols);
  rows = malloc((size_t)n * sizeof *rows);
  matrix_values = malloc((size_t)(nnz + n) * sizeof *matrix_values);
  last_column = malloc((size_t)n * sizeof *last_column);
  input_values = malloc((size_t)(n + 1) * sizeof *input_values);
  if (!raw_rowptr || !raw_cols || !raw_values || !b0 || !b1 || !answer || !row_nnz ||
      !cols || !rows || !matrix_values || !last_column || !input_values)
    die("out of memory");

  read_exact(raw_rowptr, (size_t)(n + 1) * sizeof *raw_rowptr);
  read_exact(raw_cols, (size_t)nnz * sizeof *raw_cols);
  read_exact(raw_values, (size_t)nnz * sizeof *raw_values);

  for (int i = 0; i < n; ++i) {
    row_nnz[i] = (PetscInt)(raw_rowptr[i + 1] - raw_rowptr[i] + 1);
    rows[i] = (PetscInt)i;
  }
  row_nnz[n] = 1;

  petsc_check(MatCreateSeqAIJ(PETSC_COMM_SELF, n + 1, n + 1, 0, row_nnz, &A), "MatCreateSeqAIJ");
  petsc_check(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE), "MatSetOption");
  for (int i = 0; i < n; ++i) {
    PetscInt count = 0;
    for (uint64_t k = raw_rowptr[i]; k < raw_rowptr[i + 1]; ++k) {
      cols[count] = (PetscInt)raw_cols[k];
      matrix_values[count++] = (PetscScalar)(-t * raw_values[k]);
    }
    cols[count] = (PetscInt)n;
    matrix_values[count++] = 0.0;
    petsc_check(MatSetValues(A, 1, &rows[i], count, cols, matrix_values, INSERT_VALUES),
                "MatSetValues(operator)");
  }
  {
    PetscInt last = (PetscInt)n;
    PetscScalar zero = 0.0;
    petsc_check(MatSetValue(A, last, last, zero, INSERT_VALUES), "MatSetValue(last row)");
  }
  petsc_check(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
  petsc_check(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");

  petsc_check(MatCreateVecs(A, &input, &output), "MatCreateVecs");
  petsc_check(MFNCreate(PETSC_COMM_SELF, &mfn), "MFNCreate");
  petsc_check(MFNSetOperator(mfn, A), "MFNSetOperator");
  petsc_check(MFNSetType(mfn, MFNEXPOKIT), "MFNSetType");
  petsc_check(MFNSetDimensions(mfn, requests == 16 ? KRYLOV_DIMENSION_REUSE : KRYLOV_DIMENSION_ONE_SHOT),
              "MFNSetDimensions");
  petsc_check(MFNSetTolerances(mfn, multiplier * rtol, PETSC_DETERMINE), "MFNSetTolerances");
  petsc_check(MFNGetFN(mfn, &fn), "MFNGetFN");
  petsc_check(FNSetType(fn, FNEXP), "FNSetType");

  for (int request = 0; request < requests; ++request) {
    const PetscScalar *output_values;
    read_exact(b0, (size_t)n * sizeof *b0);
    read_exact(b1, (size_t)n * sizeof *b1);
    for (int i = 0; i < n; ++i) {
      last_column[i] = (PetscScalar)(t * b1[i]);
      input_values[i] = (PetscScalar)b0[i];
    }
    input_values[n] = 1.0;
    petsc_check(MatSetValues(A, n, rows, 1, &(PetscInt){n}, last_column, INSERT_VALUES),
                "MatSetValues(last column)");
    petsc_check(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin(update)");
    petsc_check(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd(update)");
    petsc_check(VecPlaceArray(input, input_values), "VecPlaceArray");
    petsc_check(MFNSolve(mfn, input, output), "MFNSolve");
    petsc_check(VecResetArray(input), "VecResetArray");
    petsc_check(VecGetArrayRead(output, &output_values), "VecGetArrayRead");
    for (int i = 0; i < n; ++i) answer[i] = (double)PetscRealPart(output_values[i]);
    petsc_check(VecRestoreArrayRead(output, &output_values), "VecRestoreArrayRead");
    write_all(answer, (size_t)n * sizeof *answer);
  }

  petsc_check(MFNDestroy(&mfn), "MFNDestroy");
  petsc_check(VecDestroy(&input), "VecDestroy(input)");
  petsc_check(VecDestroy(&output), "VecDestroy(output)");
  petsc_check(MatDestroy(&A), "MatDestroy");
  free(raw_rowptr);
  free(raw_cols);
  free(raw_values);
  free(b0);
  free(b1);
  free(answer);
  free(row_nnz);
  free(cols);
  free(rows);
  free(matrix_values);
  free(last_column);
  free(input_values);
  petsc_check(SlepcFinalize(), "SlepcFinalize");
  return 0;
}
