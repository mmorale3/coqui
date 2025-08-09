#ifndef UTILITIES_MATH_H
#define UTILITIES_MATH_H

namespace utils
{

/** factorial of num
 * @param num integer to be factored
 * @return num!
 *     
 * num! = 1\cdot 2\cdot ... \cdot num-1 \cdot num\f$
 */
template<typename T>
T Factorial(T num);

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
T DFactorial(T num);

}

#endif
