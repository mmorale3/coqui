#ifndef MEANFIELD_MF_SOURCE_HPP 
#define MEANFIELD_MF_SOURCE_HPP 

namespace mf 
{ 

enum mf_source_e {
  bdft_source, qe_source, pyscf_source, model_source
};

inline std::string mf_source_enum_to_string(int mf_enum) {
  switch(mf_enum) {
    case bdft_source:
      return "bdft";
    case qe_source:
      return "qe";
    case pyscf_source:
      return "pyscf";
    case model_source:
      return "model";
    default:
      return "not recognized...";
  }
}

inline mf_source_e string_to_mf_source_enum(std::string mf_source) {
  if (mf_source == "bdft_source") {
    return bdft_source;
  } else if (mf_source == "qe_source") {
    return qe_source;
  } else if (mf_source == "pyscf_source") {
    return pyscf_source;
  } else if (mf_source == "model_source") {
    return model_source;
  } else {
    utils::check(false, "Unrecognized mf source: {}. "
                        "Available options: bdft_source, qe_source, pyscf_source, model_source", mf_source);
    return bdft_source;
  }
}

enum mf_input_file_type_e {
  h5_input_type, xml_input_type 
};

}

#endif
