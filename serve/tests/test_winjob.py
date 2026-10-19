"""serve/tests/test_winjob.py - job object kills the child when the parent's handle closes."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

PARENT = f"""
import subprocess, sys, time
sys.path.insert(0, {str(ROOT)!r})
from serve.engine.winjob import contain
code = ("import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "print(g.pid, flush=True); time.sleep(120)")
child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
ok = contain(child)
print(ok, child.pid, child.stdout.readline().strip(), flush=True)
time.sleep(120)
"""


def running(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


@unittest.skipUnless(os.name == "nt", "job objects are Windows-only")
class TestWinJob(unittest.TestCase):
    def test_children_end_with_a_killed_server(self):
        python = getattr(sys, "_base_executable", None) or sys.executable
        parent = subprocess.Popen([python, "-c", PARENT], stdout=subprocess.PIPE, text=True)
        child = grandchild = None
        try:
            ok, child, grandchild = parent.stdout.readline().split()
            child, grandchild = int(child), int(grandchild)
            self.assertEqual(ok, "True")
            self.assertTrue(running(child) and running(grandchild))
            parent.kill()
            parent.wait(10)
            deadline = time.monotonic() + 10
            while (running(child) or running(grandchild)) and time.monotonic() < deadline:
                time.sleep(0.2)
            self.assertFalse(running(child), "the child outlived the server")
            self.assertFalse(running(grandchild), "the grandchild outlived the server")
        finally:
            if parent.poll() is None:
                parent.kill()
            for pid in (child, grandchild):
                if pid and running(pid):
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)

    def test_elsewhere_or_without_a_process_it_is_a_no_op(self):
        from serve.engine.winjob import contain
        self.assertFalse(contain(None))


@unittest.skipUnless(os.name == "nt", "power throttling is a Windows setting")
class TestNoThrottle(unittest.TestCase):
    def test_a_contained_child_is_opted_out_of_power_throttling(self):
        from serve.engine import winjob
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(winjob.no_throttle(int(child._handle)))
            self.assertTrue(winjob.contain(child))
        finally:
            child.kill()
            child.wait(10)


if __name__ == "__main__":
    unittest.main()
