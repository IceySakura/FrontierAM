#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>

template <int K>
static double advance(long n, double *x, double *r, double *p,
                      const double *q, const double *u, const double *v,
                      double rr, double *gain) {
    double pq = 0.0;
    #pragma omp simd reduction(+:pq)
    for (long i = 0; i < n; ++i) pq += p[i] * q[i];
    if (!(pq > 0.0) || !std::isfinite(pq) || !(rr > 0.0)) return -1.0;
    const double alpha = rr / pq;
    double rr2 = 0.0;
    #pragma omp simd reduction(+:rr2)
    for (long i = 0; i < n; ++i) {
        x[i] += alpha * p[i];
        r[i] -= alpha * q[i];
        rr2 += r[i] * r[i];
    }
    double coefficients[K ? K : 1];
    for (int j = 0; j < K; ++j) {
        const double *col = v + j * n;
        double a = 0.0;
        #pragma omp simd reduction(+:a)
        for (long i = 0; i < n; ++i) a += col[i] * r[i];
        coefficients[j] = a;
    }
    const double beta = rr2 / rr;
    #pragma omp simd
    for (long i = 0; i < n; ++i) {
        double z = r[i] + beta * p[i];
        for (int j = 0; j < K; ++j) z -= u[j * n + i] * coefficients[j];
        p[i] = z;
    }
    *gain = alpha * rr;
    return rr2;
}

static PyObject *step(PyObject *, PyObject *args) {
    PyObject *ox, *orr, *op, *oq, *ou, *ov;
    double rr;
    if (!PyArg_ParseTuple(args, "OOOOOOd", &ox, &orr, &op, &oq, &ou, &ov, &rr))
        return nullptr;
    auto *ax = reinterpret_cast<PyArrayObject *>(ox);
    long n = static_cast<long>(PyArray_SIZE(ax));
    auto *x = static_cast<double *>(PyArray_DATA(ax));
    auto *r = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(orr)));
    auto *p = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(op)));
    auto *q = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(oq)));
    int k = 0;
    double *u = nullptr, *v = nullptr;
    if (ou != Py_None) {
        auto *au = reinterpret_cast<PyArrayObject *>(ou);
        k = static_cast<int>(PyArray_DIM(au, 1));
        u = static_cast<double *>(PyArray_DATA(au));
        v = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(ov)));
    }
    double result = -1.0, gain = 0.0;
    switch (k) {
        case 0: result = advance<0>(n, x, r, p, q, u, v, rr, &gain); break;
        case 1: result = advance<1>(n, x, r, p, q, u, v, rr, &gain); break;
        case 2: result = advance<2>(n, x, r, p, q, u, v, rr, &gain); break;
        case 3: result = advance<3>(n, x, r, p, q, u, v, rr, &gain); break;
        case 4: result = advance<4>(n, x, r, p, q, u, v, rr, &gain); break;
        case 5: result = advance<5>(n, x, r, p, q, u, v, rr, &gain); break;
        case 6: result = advance<6>(n, x, r, p, q, u, v, rr, &gain); break;
        case 7: result = advance<7>(n, x, r, p, q, u, v, rr, &gain); break;
        case 8: result = advance<8>(n, x, r, p, q, u, v, rr, &gain); break;
        default:
            PyErr_SetString(PyExc_ValueError, "recycling rank must not exceed 8");
            return nullptr;
    }
    return Py_BuildValue("(dd)", result, gain);
}

static PyMethodDef methods[] = {
    {"step", step, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "spd_core", nullptr, -1, methods};
PyMODINIT_FUNC PyInit_spd_core(void) {
    import_array();
    return PyModule_Create(&module);
}
