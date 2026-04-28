#!/usr/bin/env python3
"""
Patch pro_scripts/vc4_bundle_autorun.py so Ctrl-C kills a browser/Codex/test
child process that was started in its own session/process group.

The autorunner intentionally starts child commands in their own session so it can
kill the whole process group on timeout. On macOS/Linux, that also means a
terminal Ctrl-C reaches the Python parent but not the child. Without this patch,
interrupting the driver can leave new_tab.js/current_tab.js alive in the
background, and that orphan can later paste/submit a prompt after the user
thought the run was stopped.
"""

from pathlib import Path

path = Path("pro_scripts/vc4_bundle_autorun.py")
text = path.read_text()

marker = "stage={stage} received KeyboardInterrupt; terminating child process group"
if marker in text:
    print("vc4_bundle_autorun.py already has KeyboardInterrupt child cleanup")
    raise SystemExit(0)

old = """        try:\n            exit_code = proc.wait(timeout=timeout_sec)\n        except subprocess.TimeoutExpired:\n            timed_out = True\n"""

new = """        try:\n            exit_code = proc.wait(timeout=timeout_sec)\n        except KeyboardInterrupt:\n            log.write(f\"\\n[INTERRUPT] stage={stage} received KeyboardInterrupt; terminating child process group\\n\")\n            log.flush()\n            if os.name != \"nt\":\n                try:\n                    os.killpg(proc.pid, signal.SIGTERM)\n                except ProcessLookupError:\n                    pass\n            else:\n                proc.terminate()\n            try:\n                proc.wait(timeout=5)\n            except subprocess.TimeoutExpired:\n                if os.name != \"nt\":\n                    try:\n                        os.killpg(proc.pid, signal.SIGKILL)\n                    except ProcessLookupError:\n                        pass\n                else:\n                    proc.kill()\n                proc.wait()\n            raise\n        except subprocess.TimeoutExpired:\n            timed_out = True\n"""

if old not in text:
    raise SystemExit(
        "Could not find the expected proc.wait block in pro_scripts/vc4_bundle_autorun.py. "
        "Paste that file before requesting an automatic patch, or patch run_command manually."
    )

path.write_text(text.replace(old, new, 1))
print("patched pro_scripts/vc4_bundle_autorun.py KeyboardInterrupt child cleanup")
