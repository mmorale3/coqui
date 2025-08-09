#ifndef METHODS_ERI_ERI_FORMAT_E_HPP
#define METHODS_ERI_ERI_FORMAT_E_HPP

namespace methods
{

/***********************************************************************/
/*                           eri_storage_e                             */ 
/***********************************************************************/
enum eri_storage_e {
  incore, outcore
};

inline std::string eriform_enum_to_string(int storage_enum) {
  switch(storage_enum) {
    case eri_storage_e::incore:
      return "incore";
    case eri_storage_e::outcore:
      return "outcore";
    default:
      return "not recognized...";
  }
}

inline eri_storage_e string_to_eri_storage_enum(std::string eriform) {
  if (eriform == "incore") {
    return eri_storage_e::incore;
  } else if (eriform == "outcore") {
    return eri_storage_e::outcore;
  } else {
    utils::check(false, "Unrecognized storage type: {}. Available options: incore, outcore", eriform);
    return eri_storage_e::incore;
  }
}

}

#endif // METHODS_ERI_ERI_FORMAT_E_HPP
