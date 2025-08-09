#ifndef COQUI_VECTOR_SPACE_H
#define COQUI_VECTOR_SPACE_H

namespace iter_scf {

// Abstract vector spaces and operations used in DIIS
// Vectors are stored in the external file
// The class must be initialized before usage

template<typename Vector>
class VSpace {
private:
    size_t _size; // Subspace size
    std::string _filename; // name of the file containing the vector space
    bool inited = false;

public:

    VSpace() {
        _size = 0;
    }

    VSpace(std::string filename) : _filename(filename) {
        _size = 0;
        inited = true;
    }

    void initialize(std::string filename) {
        if(!inited) {
            _filename = filename;
            inited = true;
        }
    }

    Vector get_vec(const size_t i) {
        utils::check(inited, "VSpace is not initialized");
        utils::check(i < _size, "VSpace::get_vec Vector index of the VSpace container {} is out of bounds", _filename);
        Vector vec;
        vec.read_from_file(_filename, i);
        return vec;
    };

    void get_vec(const size_t i, Vector& vec) {
        utils::check(inited, "VSpace is not initialized");
        utils::check(i < _size, "VSpace::get_vec Vector index of the VSpace container {} is out of bounds", _filename);
        vec.read_from_file(_filename, i);
    }

    void add_to_vspace(Vector& a) {
        utils::check(inited, "VSpace is not initialized");
        a.write_to_file(_filename, _size);
        _size++;
    }

    std::complex<double> overlap(const size_t i, const size_t j) {
        utils::check(inited, "VSpace is not initialized");
        utils::check(i < _size, "VSpace::overlap Vector index of the VSpace container {} is out of bounds", _filename);
        utils::check(j < _size, "VSpace::overlap Vector index of the VSpace container {} is out of bounds", _filename);
        Vector vec_i;
        vec_i.read_from_file(_filename, i);
        Vector vec_j;
        vec_j.read_from_file(_filename, j);
        return overlap(vec_i, vec_j);
    }

    std::complex<double> overlap(const size_t i, const Vector& a) {
        utils::check(inited, "VSpace is not initialized");
        utils::check(i < _size, "VSpace::overlap Vector index of the VSpace container {} is out of bounds", _filename);
        Vector vec_i;
        vec_i.read_from_file(_filename, i);
        return overlap(vec_i, a);
    }

    std::complex<double> overlap(const Vector& a, const Vector& b) {
        return a.dot_prod(b);
    }

    size_t size() {
        return _size; 
    };

    // TODO: implement through move
    void purge_vec(const size_t k) {
        utils::check(inited, "VSpace is not initialized");
        utils::check(k < _size, "VSpace::purge_vec Vector index of the VSpace container {} is out of bounds", _filename);
        utils::check(_size > 0, "VSpace::purge_vec VSpace is of zero size, no vector can be deleted");
        Vector vec;
        for(size_t j = k+1; j < size(); j++) {
            vec.read_from_file(_filename, j);
            vec.write_to_file(_filename, j-1);
        }
        
        _size--; 
    }

    virtual Vector make_linear_comb(const nda::array<ComplexType, 1>& C) {
        utils::check(inited, "VSpace is not initialized");
         Vector r;
         if(_size > 0) {
             get_vec(size()-1, r); // this is needed to initialize r
             r.set_zero();
         }
         else return r; 
         for(size_t i = 0; i < _size && i < C.size(); i++) {
             ComplexType coeff = C(C.size()-1-i);
             r.add(get_vec(size()-1-i), coeff);
         }
         return r;
     }

};


} // namespace
#endif //  COQUI_VECTOR_SPACE_H
