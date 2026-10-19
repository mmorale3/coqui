// standalone shim for the rung mini-app: APP_ABORT without CoQui's logging / MPI
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#define APP_ABORT(msg) do { std::fprintf(stderr, "ABORT: %s\n", std::string(msg).c_str()); std::abort(); } while (0)
