////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

#ifndef NUMERICS_DETAIL_CPU_SPARSE_HPP
#define NUMERICS_DETAIL_CPU_SPARSE_HPP

#include "configuration.hpp"
#include "numerics/sparse/detail/CPU/sparse_cpu.hpp"

namespace math::sparse
{

template<typename T1, typename T2, typename T3, typename T4,
         typename T5, typename T6, typename T7, typename T8>
void csrmv(char transa, int M, int K, T1 alpha, const char* matdescra,
           T2 A, T3 indx, T4 pntrb, T5 pntre, T6 x, T7 beta, T8 y)
{ 
  cpu::csrmv(transa,M,K,alpha,matdescra,A,indx,pntrb,pntre,x,beta,y);
}   

template<typename T1, typename T2, typename T3, typename T4,
	 typename T5, typename T6, typename T7, typename T8>
void csrmm(char transa, int M, int N, int K, T1 alpha, const char* matdescra, 
	   T2 A, T3 indx, T4 pntrb, T5 pntre, T6 B, int ldb, int strideB, 
	   T7 beta, T8 C, int ldc, int strideC, int nbatch)
{
  auto B_ = B;
  auto C_ = C;
  for(int n=0; n<nbatch; ++n, B_+=strideB, C_+=strideC)
    cpu::csrmm_impl(transa,M,N,K,alpha,matdescra,A,indx,pntrb,pntre,B_,ldb,beta,C_,ldc);
}

} // namespace math::sparse

#endif
