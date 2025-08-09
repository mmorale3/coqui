#ifndef COQUI_MB_SOLVER_T_H
#define COQUI_MB_SOLVER_T_H

#include "methods/HF/hf_t.h"
#include "methods/embedding/projector_boson_t.h"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/GW/gw_t.h"
#include "methods/GF2/gf2_t.h"

namespace methods::solvers {

template<typename corr_solver_t = gw_t>
struct mb_solver_t {
  hf_t *hf;
  corr_solver_t *corr = nullptr;
  scr_coulomb_t *scr_eri = nullptr;

  mb_solver_t(hf_t *hf_) : hf(hf_) {}
  mb_solver_t(hf_t *hf_, corr_solver_t *corr_) : 
                           hf(hf_), corr(corr_) {}
  mb_solver_t(hf_t *hf_, corr_solver_t *corr_, scr_coulomb_t *scr_eri_) : 
                           hf(hf_), corr(corr_), scr_eri(scr_eri_) {}

  ~mb_solver_t() = default;
};

} // methods::solvers

#endif //COQUI_MB_SOLVER_T_H
