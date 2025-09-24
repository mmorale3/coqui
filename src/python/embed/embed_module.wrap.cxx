/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */



// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API
#ifndef CLAIR_C2PY_WRAP_GEN
#ifdef __clang__
// #pragma clang diagnostic ignored "-W#warnings"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wcpp"
#endif

#define C2PY_VERSION_MAJOR 0
#define C2PY_VERSION_MINOR 1

#include <c2py/c2py.hpp>

using c2py::operator""_a;

// ==================== Wrapped classes =====================

// ==================== enums =====================

// ==================== module classes =====================

// ==================== module functions ====================

// dmft_embed
static auto const fun_0 = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](const coqui_py::Mf &mf, const std::string &embed_params,
           const nda::array<ComplexType, 5> &C_ksIai,
           const nda::array<long, 3> &band_window,
           const nda::array<double, 2> &kpts_crys,
           std::optional<std::map<std::string, nda::array<ComplexType, 4>>>
               local_hf_potentials,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_selfenergies) {
          return coqui_py::dmft_embed(mf, embed_params, C_ksIai, band_window,
                                      kpts_crys, local_hf_potentials,
                                      local_selfenergies);
        },
        "mf", "embed_params", "C_ksIai", "band_window", "kpts_crys",
        "local_hf_potentials", "local_selfenergies"),
    c2py::cfun(
        [](const coqui_py::Mf &mf, const std::string &embed_params) {
          return coqui_py::dmft_embed(mf, embed_params);
        },
        "mf", "embed_params")};

// downfold_1e
static auto const fun_1 = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](const coqui_py::Mf &mf, const std::string &df_params,
           const nda::array<ComplexType, 5> &C_ksIai,
           const nda::array<long, 3> &band_window,
           const nda::array<double, 2> &kpts_crys,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_selfenergies,
           std::optional<std::map<std::string, nda::array<ComplexType, 4>>>
               local_hf_potentials) {
          return coqui_py::downfold_1e(mf, df_params, C_ksIai, band_window,
                                       kpts_crys, local_selfenergies,
                                       local_hf_potentials);
        },
        "mf", "df_params", "C_ksIai", "band_window", "kpts_crys",
        "local_selfenergies", "local_hf_potentials"),
    c2py::cfun(
        [](const coqui_py::Mf &mf, const std::string &df_params,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_selfenergies,
           std::optional<std::map<std::string, nda::array<ComplexType, 4>>>
               local_hf_potentials) {
          return coqui_py::downfold_1e(mf, df_params, local_selfenergies,
                                       local_hf_potentials);
        },
        "mf", "df_params", "local_selfenergies", "local_hf_potentials")};

// downfold_2e
static auto const fun_2 = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
           const nda::array<ComplexType, 5> &C_ksIai,
           const nda::array<long, 3> &band_window,
           const nda::array<double, 2> &kpts_crys,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_polarizabilities) {
          return coqui_py::downfold_2e(eri, df_params, C_ksIai, band_window,
                                       kpts_crys, local_polarizabilities);
        },
        "eri", "df_params", "C_ksIai", "band_window", "kpts_crys",
        "local_polarizabilities"),
    c2py::cfun(
        [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_polarizabilities) {
          return coqui_py::downfold_2e(eri, df_params, local_polarizabilities);
        },
        "eri", "df_params", "local_polarizabilities")};

// downfold_2e_return_vw
static auto const fun_3 = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
           const nda::array<ComplexType, 5> &C_ksIai,
           const nda::array<long, 3> &band_window,
           const nda::array<double, 2> &kpts_crys,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_polarizabilities) {
          return coqui_py::downfold_2e_return_vw(eri, df_params, C_ksIai,
                                                 band_window, kpts_crys,
                                                 local_polarizabilities);
        },
        "eri", "df_params", "C_ksIai", "band_window", "kpts_crys",
        "local_polarizabilities"),
    c2py::cfun(
        [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_polarizabilities) {
          return coqui_py::downfold_2e_return_vw(eri, df_params,
                                                 local_polarizabilities);
        },
        "eri", "df_params", "local_polarizabilities")};

// downfold_gloc
static auto const fun_4 = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](const coqui_py::Mf &mf, const std::string &df_params) {
          return coqui_py::downfold_gloc(mf, df_params);
        },
        "mf", "df_params"),
    c2py::cfun(
        [](const coqui_py::Mf &mf, const std::string &df_params,
           const nda::array<ComplexType, 5> &C_ksIai,
           const nda::array<long, 3> &band_window,
           const nda::array<double, 2> &kpts_crys) {
          return coqui_py::downfold_gloc(mf, df_params, C_ksIai, band_window,
                                         kpts_crys);
        },
        "mf", "df_params", "C_ksIai", "band_window", "kpts_crys")};

// downfold_wloc
static auto const fun_5 = c2py::dispatcher_f_kw_t{
    c2py::cfun(
        [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_polarizabilities) {
          return coqui_py::downfold_wloc(eri, df_params,
                                         local_polarizabilities);
        },
        "eri", "df_params", "local_polarizabilities"),
    c2py::cfun(
        [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
           const nda::array<ComplexType, 5> &C_ksIai,
           const nda::array<long, 3> &band_window,
           const nda::array<double, 2> &kpts_crys,
           std::optional<std::map<std::string, nda::array<ComplexType, 5>>>
               local_polarizabilities) {
          return coqui_py::downfold_wloc(eri, df_params, C_ksIai, band_window,
                                         kpts_crys, local_polarizabilities);
        },
        "eri", "df_params", "C_ksIai", "band_window", "kpts_crys",
        "local_polarizabilities")};

static const auto doc_d_0 = fun_0.doc(R"DOC()DOC");
static const auto doc_d_1 = fun_1.doc(R"DOC()DOC");
static const auto doc_d_2 = fun_2.doc(R"DOC()DOC");
static const auto doc_d_3 = fun_3.doc(R"DOC()DOC");
static const auto doc_d_4 = fun_4.doc(R"DOC()DOC");
static const auto doc_d_5 = fun_5.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"dmft_embed", (PyCFunction)c2py::pyfkw<fun_0>,
     METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
    {"downfold_1e", (PyCFunction)c2py::pyfkw<fun_1>,
     METH_VARARGS | METH_KEYWORDS, doc_d_1.c_str()},
    {"downfold_2e", (PyCFunction)c2py::pyfkw<fun_2>,
     METH_VARARGS | METH_KEYWORDS, doc_d_2.c_str()},
    {"downfold_2e_return_vw", (PyCFunction)c2py::pyfkw<fun_3>,
     METH_VARARGS | METH_KEYWORDS, doc_d_3.c_str()},
    {"downfold_gloc", (PyCFunction)c2py::pyfkw<fun_4>,
     METH_VARARGS | METH_KEYWORDS, doc_d_4.c_str()},
    {"downfold_wloc", (PyCFunction)c2py::pyfkw<fun_5>,
     METH_VARARGS | METH_KEYWORDS, doc_d_5.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "embed_module",                              /* name of module */
    R"RAWDOC(Embedding module for CoQui)RAWDOC", /* module documentation, may be
                                                    NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module
           keeps state in global variables. */
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *
PyInit_embed_module() {

  if (not c2py::check_python_version("embed_module"))
    return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL)
    return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] =
      &c2py::wrap_pytype<c2py::py_range>;

  return m;
}
#endif
// CLAIR_WRAP_GEN
