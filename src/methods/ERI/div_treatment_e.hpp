#ifndef METHODS_ERI_DIV_TREATMENT_HPP
#define METHODS_ERI_DIV_TREATMENT_HPP

namespace methods
{

enum div_treatment_e {
  ignore_g0, gygi, gygi_extrplt, gygi_extrplt_2d
};

inline std::string div_enum_to_string(int div_enum) {
  switch(div_enum) {
    case div_treatment_e::ignore_g0:
      return "ignore_g0";
    case div_treatment_e::gygi:
      return "gygi";
    case div_treatment_e::gygi_extrplt:
      return "gygi_extrplt";
    case div_treatment_e::gygi_extrplt_2d:
      return "gygi_extrplt_2d";
    default:
      return "not recognized...";
  }
}

inline div_treatment_e string_to_div_enum(std::string div_name) {
  if (div_name == "ignore_g0") {
    return div_treatment_e::ignore_g0;
  } else if (div_name == "gygi") {
    return div_treatment_e::gygi;
  } else if (div_name == "gygi_extrplt") {
    return div_treatment_e::gygi_extrplt;
  } else if (div_name == "gygi_extrplt_2d") {
    return div_treatment_e::gygi_extrplt_2d;
  } else {
    utils::check(false, "Unrecognized divergence treatment: {}. "
                        "Available options: ignore_g0, gygi, gygi_extrplt, gygi_extrplt_2d \n",
                        "(The option \"ewald\" is now renamed to \"gygi\")", div_name);
    return div_treatment_e::gygi;
  }
}

}

#endif // METHODS_ERI_DIV_TREATMENT_HPP
