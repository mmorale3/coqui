
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
