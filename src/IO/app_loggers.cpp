#if defined(ENABLE_SPDLOG)
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#endif

int __app_debug_level__  = -10000; 
int __app_output_level__ = -10000; 
bool __app_stacktrace__  = true;

// currently using 2 separate loggers
// app_log: uses "std_console" with a clean format only on Global().root()
// app_warning/app_error/app_critical/app_debug: use "err_console" shared by everyone
// consider later using app_debug as a separate logger to file per mpi rank 
// MAM 07.25.24: Moving wanings to a separate console

void setup_loggers(bool root, int output_level, int debug_level)
{
  __app_debug_level__ = debug_level;
  if(root) {
    __app_output_level__ = output_level;
#if defined(ENABLE_SPDLOG)
    auto l = spdlog::get("std_console");
    if(not l) {
      auto console = spdlog::stdout_color_mt("std_console");  
      spdlog::get("std_console")->set_pattern("%v");
    }
#endif
  } else {
    __app_output_level__ = -10000;
  }
#if defined(ENABLE_SPDLOG)
  {
    auto l = spdlog::get("warn_console");
    if(not l) {
      auto warn_logger = spdlog::stdout_color_mt("warn_console");
      spdlog::get("warn_console")->set_pattern("%^[%l]%$ %v");
    }
    spdlog::get("warn_console")->set_level(spdlog::level::warn);
  }
  auto l = spdlog::get("err_console");
  if(not l) {
    auto err_logger = spdlog::stdout_color_mt("err_console");   
    spdlog::get("err_console")->set_pattern("%^[%l]%$ %v");
  }
  if(debug_level > 0)
    spdlog::get("err_console")->set_level(spdlog::level::debug);
#endif
}

void set_debug_level([[maybe_unused]] bool root, int debug_level)
{
  __app_debug_level__ = debug_level;
#if defined(ENABLE_SPDLOG)
  if(debug_level > 0) {  
    auto l = spdlog::get("err_console");
    if(not l) {
      auto err_logger = spdlog::stdout_color_mt("err_console");
      spdlog::get("err_console")->set_pattern("%^[%l]%$ %v");
    }
    spdlog::get("err_console")->set_level(spdlog::level::debug);
  }
#endif
}

void set_output_level(bool root, int output_level)
{
  if(root) {
    __app_output_level__ = output_level;
#if defined(ENABLE_SPDLOG)
    // should check that logger exists! 
    auto l = spdlog::get("std_console");
    if(not l) {
      auto console = spdlog::stdout_color_mt("std_console");
      spdlog::get("std_console")->set_pattern("%v");
    }
#endif
  } else
    __app_output_level__ = -10000;
#if defined(ENABLE_SPDLOG)
  {
    auto l = spdlog::get("warn_console");
    if(not l) {
      auto warn_logger = spdlog::stdout_color_mt("warn_console");
      spdlog::get("warn_console")->set_pattern("%^[%l]%$ %v");
    }
    spdlog::get("warn_console")->set_level(spdlog::level::warn);
  }
#endif
}

void set_stacktrace(bool stk) { __app_stacktrace__ = stk; }

