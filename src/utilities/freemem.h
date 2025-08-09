#ifndef AIMBES_UTILITIES_FREEMEM_H
#define AIMBES_UTILITIES_FREEMEM_H

#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/resource.h>
#endif

namespace utils {

std::size_t freemem();
std::size_t freemem_device();
void memory_report(int io_lvl = 3, std::string message = {});

}

#endif
