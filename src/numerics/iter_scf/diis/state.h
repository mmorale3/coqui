#ifndef COQUI_STATE_HPP
#define COQUI_STATE_HPP

namespace iter_scf {

/**
 * Representation of a state of an optimization problem
 */

template<typename Vector>
class opt_state {
    public:
    opt_state() {}
    opt_state(Vector& x_) : x(x_) {inited = true;}

    void initialize(Vector& x_) {
        if(!inited) { x = x_; }
        inited = true;
    }

    Vector get() const {
        utils::check(inited, "State is not initialized");
        return x;
    }

    void set(const Vector x_) {x = x_; inited = true;}
    void set(const Vector& x_) {x = x_; inited = true;}

    void put(const Vector& x_) {x = x_; inited = true;}

    bool is_inited() const {return inited;}
    
    private:
    Vector x;
    bool inited = false;
};

}

#endif
