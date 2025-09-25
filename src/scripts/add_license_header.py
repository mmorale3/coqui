#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# ---- Configuration ----
HEADER_CPP = """\
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
"""

HEADER_PY = '''"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""'''

HEADER_FORTRAN = '''\
!=============================================================================
! CoQuí: Correlated Quantum ínterface
!
! Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
!=============================================================================
'''


CPP_EXTENSIONS = {".cpp", ".cxx", ".cc", ".h", ".hpp", ".hh", ".cu", ".cuh"}
PY_EXTENSIONS = {".py"}
FORTRAN_EXTENSIONS = {".f90", ".f95", ".f03", ".f08", ".f", ".f77"}

# =============================================================================
# Helpers
# =============================================================================

def choose_header(ext: str) -> str | None:
    if ext in CPP_EXTENSIONS:
        return HEADER_CPP
    if ext in PY_EXTENSIONS:
        return HEADER_PY
    if ext in FORTRAN_EXTENSIONS:
        return HEADER_FORTRAN
    return None


def file_needs_header(path: Path, header: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            start = f.read(len(header))
            return "Apache License" not in start
    except Exception:
        return False


def add_header(path: Path, header: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n" + content)


def main(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = Path(filename).suffix
            header = choose_header(ext)
            if header is None:
                continue
            path = Path(dirpath) / filename
            if file_needs_header(path, header):
                print(f"Adding header to {path}")
                add_header(path, header)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: add_license_headers.py <project_root>")
        sys.exit(1)
    main(Path(sys.argv[1]))
