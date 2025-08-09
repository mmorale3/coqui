#include <string>
#include <iostream>
#include <fstream>
#include <vector>

namespace utils
{

std::vector<std::string> split(std::string const& str, std::string const& delim)
{
  std::vector<std::string> w;
  auto beg = str.find_first_not_of(delim);
  while(beg != std::string::npos) {
    auto end=str.find_first_of(delim, beg+1);
    if(end == std::string::npos) {
      w.emplace_back(str.substr(beg,str.size()-beg)); 
      break;
    }
    w.emplace_back(str.substr(beg,end-beg)); 
    beg = str.find_first_not_of(delim,end+1);
  }  
  return w;
}

}
