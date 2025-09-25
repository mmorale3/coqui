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



namespace utils
{

/** factorial of num
 * @param num integer to be factored
 * @return num!
 *     
 * num! = 1\cdot 2\cdot ... \cdot num-1 \cdot num\f$
 */  
template<typename T>
T Factorial(T num) { return (num < T(2)) ? T(1) : num * Factorial<T>(num - T(1)); }

/**  double factorial of num
 * @param num integer to be factored
 * @return num!!
 *     
 * \if num == odd,
 * \f$ num!! = 1\cdot 3\cdot ... \cdot num-2 \cdot num\f$
 * \else num == even,
 * \f$ num!! = 2\cdot 4\cdot ... \cdot num-2 \cdot num\f$
 */  
template<typename T>
T DFactorial(T num) { return (num < T(2)) ? T(1) : num * DFactorial<T>(num - T(2)); }


template int Factorial(int);
template long Factorial(long);
template int DFactorial(int);
template long DFactorial(long);

}
