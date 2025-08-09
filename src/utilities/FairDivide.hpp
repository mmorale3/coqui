//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


#include <cstdlib>
#include <tuple>
#include <algorithm>

#ifndef UTILITIES_FAIRDIVIDE_HPP
#define UTILITIES_FAIRDIVIDE_HPP

template<typename IType>
inline std::tuple<IType, IType> FairDivideBoundary(IType me, IType ntot, IType npart)
{
  IType bat     = ntot / npart;
  IType residue = ntot % npart;
  if (me < residue)
    return std::make_tuple(me * (bat + 1), (me + 1) * (bat + 1));
  else
    return std::make_tuple(me * bat + residue, (me + 1) * bat + residue);
}

template<class IVec>
inline void FairDivide(int ntot, int npart, IVec& adist)
{
  if (adist.size() != npart + 1)
    adist.resize(npart + 1);
  int bat     = ntot / npart;
  int residue = ntot % npart;
  adist[0]    = 0;
  for (int i = 1; i < npart; i++)
  {
    if (i < residue)
      adist[i] = adist[i - 1] + bat + 1;
    else
      adist[i] = adist[i - 1] + bat;
  }
  adist[npart] = ntot;
}

#endif
