#!/usr/bin/env python3
"""Runs the metal-cpp on Asahi Linux test programs and summarizes them.

  run-tests.py build/tests/basic build/tests/compute ... [--filter NAME] [--timeout S]

Each program prints PASS/FAIL lines per case and exits non-zero on failure.
METAL_ASAHI_MODE selects hw | sim | preflight | verify (see README.md). In
verify mode a GPU/simulator mismatch is reported by the runtime on stderr
("verify: ... differ"), and counts as a failure here.
"""
import os
import subprocess
import sys
import time


def main():
    args = sys.argv[1:]
    flt = None
    timeout = float(os.environ.get("TEST_TIMEOUT", "900"))
    progs = []
    i = 0
    while i < len(args):
        if args[i] == "--filter":
            flt = args[i + 1]
            i += 2
        elif args[i] == "--timeout":
            timeout = float(args[i + 1])
            i += 2
        else:
            progs.append(args[i])
            i += 1
    mode = os.environ.get("METAL_ASAHI_MODE", "preflight")
    total = passed = 0
    failed = []
    t0 = time.time()
    for p in progs:
        cmd = [p] + ([flt] if flt else [])
        start = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            out, err, rc = r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = "timeout after %.0f s" % timeout
            rc = -1
        name = os.path.basename(p)
        cases = [l for l in out.splitlines() if l.startswith(("PASS:", "FAIL:"))]
        mism = [l for l in err.splitlines() if "verify:" in l and "differ" in l]
        n_pass = sum(1 for l in cases if l.startswith("PASS:"))
        n_fail = sum(1 for l in cases if l.startswith("FAIL:"))
        total += len(cases)
        passed += n_pass
        status = "ok" if rc == 0 and not mism else "FAILED"
        print("%-24s %3d/%-3d passed  %6.1f s  %s" % (name, n_pass, len(cases), time.time() - start, status))
        for l in cases:
            if l.startswith("FAIL:"):
                failed.append("%s: %s" % (name, l[6:]))
        if mism:
            failed.append("%s: %d GPU/simulator mismatch(es)" % (name, len(mism)))
        if rc != 0 and n_fail == 0:
            failed.append("%s: exited with %d" % (name, rc))
        if status != "ok":
            tail = (err.strip().splitlines() or [""])[-12:]
            for l in tail:
                print("    | " + l)
    print("\nmode %s: %d/%d cases passed in %.1f s" % (mode, passed, total, time.time() - t0))
    if failed:
        print("failures:")
        for f in failed:
            print("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
