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
