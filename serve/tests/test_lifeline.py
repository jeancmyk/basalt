"""serve/engine/lifeline.py: the engine ends with the server, however the server ends."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.engine import lifeline

ROOT = Path(__file__).resolve().parents[2]

SERVER = textwrap.dedent("""
    import json, os, subprocess, sys, threading, time
    sys.path.insert(0, {root!r})
    from serve.engine import lifeline
    sleeper = [sys.executable, "-c", "import time; time.sleep(600)"]
    box = []
    if sys.argv[1] == "thread":                 # started on a request thread that ends at once
        t = threading.Thread(target=lambda: box.append(lifeline.popen(sleeper)))
        t.start(); t.join()
    else:
        box.append(lifeline.popen(sleeper))
    print(json.dumps({{"server": os.getpid(), "child": box[0].pid, "group": os.getpgid(box[0].pid),
                       "server_group": os.getpgid(0)}}), flush=True)
    time.sleep(600)
""")


class Flags(unittest.TestCase):
    """What the Popen gets on each platform, without starting anything."""

    def test_windows_gets_a_process_group_and_a_job(self):
        with mock.patch.object(lifeline, "WINDOWS", True), mock.patch.object(lifeline, "_contain") as contain, \
                mock.patch.object(subprocess, "Popen") as popen:
            lifeline.popen(["x"], creationflags=0x8)
        self.assertEqual(popen.call_args.kwargs["creationflags"],
                         0x8 | lifeline.NEW_PROCESS_GROUP | lifeline.BREAKAWAY_FROM_JOB)
        self.assertNotIn("preexec_fn", popen.call_args.kwargs)
        contain.assert_called_once_with(popen.return_value)

    def test_a_job_without_breakaway_gets_one_more_spawn_without_the_flag(self):
        """A supervisor's job fails CREATE_BREAKAWAY_FROM_JOB with ACCESS_DENIED; the second spawn drops it."""
        denied = OSError(5, "Access is denied")
        denied.winerror = 5
        calls = []

        def fake(args, **kw):
            calls.append(dict(kw))
            if len(calls) == 1:
                raise denied
            return mock.Mock()

        with mock.patch.object(lifeline, "WINDOWS", True), mock.patch.object(lifeline, "_contain"), \
                mock.patch.object(subprocess, "Popen", fake):
            lifeline.popen(["x"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["creationflags"] & lifeline.BREAKAWAY_FROM_JOB)
        self.assertEqual(calls[1]["creationflags"],
                         lifeline.NEW_PROCESS_GROUP)
        self.assertTrue(calls[1]["creationflags"] & lifeline.NEW_PROCESS_GROUP)

    def test_any_other_spawn_failure_propagates(self):
        missing = OSError(2, "No such file")
        missing.winerror = 2
        with mock.patch.object(lifeline, "WINDOWS", True), mock.patch.object(lifeline, "_contain"), \
                mock.patch.object(subprocess, "Popen", side_effect=missing) as popen:
            with self.assertRaises(OSError):
                lifeline.popen(["x"])
        popen.assert_called_once()       # no retry for anything but the breakaway refusal

    def test_posix_starts_from_the_anchor_thread_with_a_child_setup(self):
        seen = {}

        def fake(args, **kw):
            seen["thread"], seen["kw"] = threading.current_thread().name, kw
            return mock.Mock()

        with mock.patch.object(lifeline, "WINDOWS", False), mock.patch.object(subprocess, "Popen", fake):
            lifeline.popen(["x"], cwd="/")
        self.assertEqual(seen["thread"], "engine-anchor")
        self.assertNotEqual(seen["thread"], threading.current_thread().name)
        self.assertEqual(seen["kw"]["cwd"], "/")
        self.assertNotIn("creationflags", seen["kw"])
        self.assertTrue(callable(seen["kw"]["preexec_fn"]))

    @unittest.skipUnless(hasattr(os, "setpgid"), "needs POSIX process groups")
    def test_the_child_setup_makes_a_group_sets_the_death_signal_and_chains(self):
        calls = []
        then = mock.Mock()
        with mock.patch.object(os, "setpgid", lambda *a: calls.append(("setpgid", a))), \
                mock.patch.object(lifeline, "_prctl", lambda *a: calls.append(("prctl", a))), \
                mock.patch.object(os, "getppid", return_value=77):
            lifeline._child_setup(77, then)()
        self.assertEqual(calls, [("setpgid", (0, 0)),
                                 ("prctl", (lifeline.PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0))])
        then.assert_called_once_with()

    @unittest.skipUnless(hasattr(os, "setpgid"), "needs POSIX process groups")
    def test_a_parent_that_already_died_kills_the_child(self):
        with mock.patch.object(os, "setpgid"), mock.patch.object(lifeline, "_prctl"), \
                mock.patch.object(os, "getppid", return_value=1), mock.patch.object(os, "kill") as kill:
            lifeline._child_setup(77)()
        kill.assert_called_once_with(os.getpid(), signal.SIGKILL)

    def test_the_job_object_is_a_no_op_off_windows(self):
        if os.name == "nt":
            self.skipTest("Windows")
        self.assertFalse(lifeline._contain(mock.Mock()))


@unittest.skipUnless(lifeline.LINUX and lifeline._prctl is not None, "needs PR_SET_PDEATHSIG")
class ServerKilled(unittest.TestCase):
    """A real server process starts a sleeping stand-in the same way the engine is started, then is SIGKILLed."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.script = Path(self.temp.name) / "server.py"
        self.script.write_text(SERVER.format(root=str(ROOT)), encoding="utf-8")
        self.pids = []
        self.addCleanup(self._reap)

    def _reap(self):
        for pid in self.pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def launch(self, mode):
        proc = subprocess.Popen([sys.executable, str(self.script), mode], stdout=subprocess.PIPE, text=True)
        info = json.loads(proc.stdout.readline())
        self.pids += [info["server"], info["child"]]
        self.assertTrue(self.alive(info["child"]))
        return proc, info

    @staticmethod
    def alive(pid):
        try:
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            return False

    def gone_within(self, pid, seconds=3.0):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if not self.alive(pid):
                return True
            time.sleep(0.05)
        return False

    def test_the_child_dies_when_the_server_is_killed(self):
        for mode in ("main", "thread"):
            with self.subTest(mode=mode):
                proc, info = self.launch(mode)
                proc.kill()
                proc.wait()
                self.assertTrue(self.gone_within(info["child"]), "the child outlived the server")

    def test_the_child_has_its_own_process_group(self):
        proc, info = self.launch("main")
        self.assertEqual(info["group"], info["child"])
        self.assertNotEqual(info["group"], info["server_group"])
        proc.kill()
        proc.wait()
