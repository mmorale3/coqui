#ifndef COQUI_QP_CONTEXT_H
#define COQUI_QP_CONTEXT_H

namespace methods {

struct qp_context_t {
  std::string qp_type = "sc";
  std::string ac_alg = "pade";
  int Nfit = 18;
  double eta = 0.0001;
  double tol = 1e-8;

  // off-diagonal mode defined in T. Kotani et. al., Phys. Rev. B 76, 165106 (2007)
  std::string off_diag_mode = "fermi";
};

} // methods

#endif //COQUI_QP_CONTEXT_H
