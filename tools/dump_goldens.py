#!/usr/bin/env python3
"""Pack .npz goldens produced by ref_gigaam.py into the flat binary container
read by tools/test_core.cpp (tools/golden.h).

Usage: tools/dump_goldens.py IN.npz OUT.bin
"""
import struct
import sys

import numpy as np

DTYPES = {"float32": 0, "float64": 1, "int64": 2, "int32": 3}


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    d = np.load(src)
    names = sorted(d.files)
    with open(dst, "wb") as f:
        f.write(b"ASRGOLD1")
        f.write(struct.pack("<I", len(names)))
        for name in names:
            a = np.ascontiguousarray(d[name])
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", a.ndim))
            for dim in a.shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", DTYPES[str(a.dtype)]))
            f.write(struct.pack("<Q", a.nbytes))
            f.write(a.tobytes())
    print(f"wrote {dst}: {len(names)} arrays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
