
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/div_treatment_e.hpp"
#include "methods/GW/gw_t.h"

namespace methods {
  namespace solvers {

    gw_t::gw_t(const imag_axes_ft::IAFT *ft, div_treatment_e div, std::string output):
       _ft(ft),
       _div_treatment(div),
       _output(output),
       _Timer() {}

    void gw_t::print_thc_gw_timers() {
      app_log(1, "\n  THC-GW timers");
      app_log(1, "  -------------");
      app_log(1, "    Total:                   {0:.3f} sec", _Timer.elapsed("TOTAL"));
      if (_Timer.elapsed("EVALUATE_PI_K") > 0.0) {
        app_log(1, "    Evaluate Pi (k):         {0:.3f} sec", _Timer.elapsed("EVALUATE_PI_K"));
        app_log(1, "      - Alloc:               {0:.3f} sec", _Timer.elapsed("PI_ALLOC_K"));
        app_log(1, "      - Gij -> Guv:          {0:.3f} sec", _Timer.elapsed("PI_PRIM_TO_AUX"));
        app_log(1, "      - Hadamard product:    {0:.3f} sec", _Timer.elapsed("PI_HADPROD_K"));
      } else if (_Timer.elapsed("EVALUATE_PI_R") > 0.0) {
        app_log(1, "    Evaluate Pi (R):         {0:.3f} sec", _Timer.elapsed("EVALUATE_PI_R"));
        app_log(1, "      - Alloc:               {0:.3f} sec", _Timer.elapsed("PI_ALLOC_R"));
        app_log(1, "      - Gij -> Guv:          {0:.3f} sec", _Timer.elapsed("PI_PRIM_TO_AUX"));
        app_log(1, "      - FT:                  {0:.3f} sec", _Timer.elapsed("PI_FT_R"));
        app_log(1, "      - Hadamard product:    {0:.3f} sec", _Timer.elapsed("PI_HADPROD_R"));
      }
      app_log(1, "    Evaluate W:              {0:.3f} sec", _Timer.elapsed("EVALUATE_W"));
      if (_Timer.elapsed("EVALUATE_SIGMA_K") > 0.0) {
        app_log(1, "    Evaluate Sigma (K):      {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA_K"));
        app_log(1, "      - Alloc:               {0:.3f} sec", _Timer.elapsed("SIGMA_ALLOC_K"));
        app_log(1, "      - Gij -> Guv:          {0:.3f} sec", _Timer.elapsed("SIGMA_PRIM_TO_AUX"));
        app_log(1, "      - Hadamard product:    {0:.3f} sec", _Timer.elapsed("SIGMA_HADPROD_K"));
        app_log(1, "      - Sigma_uv -> Sigma_ij {0:.3f} sec", _Timer.elapsed("SIGMA_AUX_TO_PRIM"));
        app_log(1, "      - Multiply dmat:       {0:.3f} sec", _Timer.elapsed("SIGMA_MULTIPLY_DMAT_K"));
      } else if (_Timer.elapsed("EVALUATE_SIGMA_R") > 0.0) {
        app_log(1, "    Evaluate Sigma (R):      {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA_R"));
        app_log(1, "      - Alloc:               {0:.3f} sec", _Timer.elapsed("SIGMA_ALLOC_R"));
        app_log(1, "      - Gij -> Guv:          {0:.3f} sec", _Timer.elapsed("SIGMA_PRIM_TO_AUX"));
        app_log(1, "      - FT:                  {0:.3f} sec", _Timer.elapsed("SIGMA_FT_R"));
        app_log(1, "      - Hadamard product:    {0:.3f} sec", _Timer.elapsed("SIGMA_HADPROD_R"));
        app_log(1, "      - Sigma_uv -> Sigma_ij {0:.3f} sec", _Timer.elapsed("SIGMA_AUX_TO_PRIM"));
      }
      app_log(1, "    Imaginary FT tau->w:     {0:.3f} sec", _Timer.elapsed("IMAG_FT_TtoW"));
      app_log(1, "    Imaginary FT w->tau:     {0:.3f} sec", _Timer.elapsed("IMAG_FT_WtoT"));
      app_log(1, "      - FT_REDISTRIBUTE:     {0:.3f} sec\n", _Timer.elapsed("FT_REDISTRIBUTE"));
    }

    void gw_t::print_thc_rpa_timers() {
      app_log(1, "\n  THC-RPA timers");
      app_log(1, "  --------------");
      app_log(1, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
      if (_Timer.elapsed("EVALUATE_PI_K") > 0.0) {
        app_log(1, "    Evaluate Pi (k):       {0:.3f} sec", _Timer.elapsed("EVALUATE_PI_K"));
        app_log(1, "      - Alloc:             {0:.3f} sec", _Timer.elapsed("PI_ALLOC_K"));
        app_log(1, "      - Gij -> Guv:        {0:.3f} sec", _Timer.elapsed("PI_PRIM_TO_AUX"));
        app_log(1, "      - Hadamard product:  {0:.3f} sec", _Timer.elapsed("PI_HADPROD_K"));
      } else if (_Timer.elapsed("EVALUATE_PI_R") > 0.0) {
        app_log(1, "    Evaluate Pi (R):       {0:.3f} sec", _Timer.elapsed("EVALUATE_PI_R"));
        app_log(1, "      - Alloc:             {0:.3f} sec", _Timer.elapsed("PI_ALLOC_R"));
        app_log(1, "      - Gij -> Guv:        {0:.3f} sec", _Timer.elapsed("PI_PRIM_TO_AUX"));
        app_log(1, "      - FT:                {0:.3f} sec", _Timer.elapsed("PI_FT_R"));
        app_log(1, "      - Hadamard product:  {0:.3f} sec", _Timer.elapsed("PI_HADPROD_R"));
      }
      app_log(1, "    Evaluate RPA:          {0:.3f} sec", _Timer.elapsed("EVALUATE_RPA"));
      app_log(1, "        - Alloc:           {0:.3f} sec", _Timer.elapsed("RPA_ALLOC"));
      app_log(1, "    Imaginary FT tau->w:   {0:.3f} sec", _Timer.elapsed("IMAG_FT_TtoW"));
      app_log(1, "      - FT_REDISTRIBUTE:   {0:.3f} sec\n", _Timer.elapsed("FT_REDISTRIBUTE"));
    }

    void gw_t::print_chol_gw_timers() {
      app_log(1, "\n  Chol-GW timers");
      app_log(1, "  --------------");
      app_log(1, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
      app_log(1, "    Allocations:           {0:.3f} sec", _Timer.elapsed("ALLOC"));
      app_log(1, "    Communication:         {0:.3f} sec", _Timer.elapsed("COMM"));
      app_log(1, "    Evaluate_P0:           {0:.3f} sec", _Timer.elapsed("EVALUATE_P0"));
      app_log(1, "    Dyson_P:               {0:.3f} sec", _Timer.elapsed("DYSON_P"));
      app_log(1, "    Evaluate_Sigma:        {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA"));
      app_log(1, "    Imaginary FT:          {0:.3f} sec", _Timer.elapsed("IMAG_FT"));
      app_log(1, "    Chol-ERI reader:       {0:.3f} sec\n", _Timer.elapsed("ERI_READER"));
    }

    void gw_t::print_rpa_gw_timers() {
      app_log(1, "\n  Chol-RPA timers");
      app_log(1, "  ---------------");
      app_log(1, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
      app_log(1, "    Allocations:           {0:.3f} sec", _Timer.elapsed("ALLOC"));
      app_log(1, "    Communication:         {0:.3f} sec", _Timer.elapsed("COMM"));
      app_log(1, "    Evaluate_P0:           {0:.3f} sec", _Timer.elapsed("EVALUATE_P0"));
      app_log(1, "    Evaluate_RPA:          {0:.3f} sec", _Timer.elapsed("EVALUATE_RPA"));
      app_log(1, "    Imaginary FT:          {0:.3f} sec", _Timer.elapsed("IMAG_FT"));
      app_log(1, "    Chol-ERI reader:       {0:.3f} sec\n", _Timer.elapsed("ERI_READER"));
    }

  } // solvers
} // methods

