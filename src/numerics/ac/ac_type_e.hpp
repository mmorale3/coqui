#ifndef COQUI_AC_TYPE_E_HPP
#define COQUI_AC_TYPE_E_HPP

namespace analyt_cont {
  enum ac_type_e {
    pade
  };

  inline std::string ac_enum_to_string(int ac_enum) {
    switch(ac_enum) {
      case ac_type_e::pade:
        return "pade";
      default:
        return "not recognized...";
    }
  }

  inline ac_type_e string_to_ac_enum(std::string ac_type) {
    if (ac_type == "pade") {
      return ac_type_e::pade;
    } else {
      utils::check(false, "Unrecognized ac_type");
      return ac_type_e::pade;
    }
  }

}

#endif //COQUI_AC_TYPE_E_HPP
