#ifndef COQUI_DIIS_RESIDUAL_H
#define COQUI_DIIS_RESIDUAL_H

#include "vspace.h"
#include "state.h"

namespace iter_scf {

// Fallback implementation of the residual as a difference between vectors
// Not tested well.

template<typename Vector>
class diis_residual {
protected:
    VSpace<Vector>* x_vsp = nullptr;   // The subspace of X vectors
    opt_state<Vector>* state = nullptr;
    bool is_initialized = false;

public:
    
    diis_residual() {} // must be initialized!

    diis_residual(VSpace<Vector>* x_space, opt_state<Vector>* state_) {
        init(x_space, state_);
    }

    virtual void init(VSpace<Vector>* x_space, opt_state<Vector>* state_) {
        x_vsp = x_space;
        state = state_;
        is_initialized = true;
    }

    bool is_inited() const {
        return is_initialized;
    }


    // Canonical implementation of the residual 
    // as a difference between successive iterations. 
    // Should be a reasonable default choice for residual definition.
    virtual bool get_diis_residual(Vector& res) {
        utils::check(is_initialized, "DIIS difference residual is not initialized");
        if(x_vsp->size() >= 2) {
            res = state->get();
            res.add(x_vsp->get_vec(x_vsp->size()-1), -1.0);
            return true;
        }
        else {
            return false;
        }
    };


};
}


#endif // COQUI_DIIS_RESIDUAL_H
