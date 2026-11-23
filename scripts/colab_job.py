"""Self-cleaning runner for Colab GPU jobs with Drive-mount handling and teardown."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional

COLAB_PYTHON = "/Users/isoamyl50/.local/share/uv/tools/google-colab-cli/bin/python"


def run_cmd(
    cmd: List[str],
    check: bool = True,
    env: Optional[dict] = None,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a CLI command and stream output."""
    print(f"[runner] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, env=env, input=input_text)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def check_drive_auth_ready(session: str, env: dict) -> bool:
    """Poll Colab credentials-propagation dryrun endpoint to see if Drive consent succeeded."""
    probe = (
        "import json; "
        "from colab_cli.auth import get_credentials; "
        "from colab_cli.common import state; "
        "from colab_cli.utils import get_status_code; "
        f"s = state.get_session({session!r}); "
        "url = f'{state.client.colab_domain}/tun/m/credentials-propagation/{s.endpoint}'; "
        "params = {'authuser':'0','authtype':'dfs_ephemeral','version':'2','dryrun':'true','propagate':'true','record':'false'}; "
        "creds = get_credentials(state.client_oauth_config, provider=state.auth_provider); "
        "r = creds.request('GET', url, params=params); "
        "tok = json.loads(r.text.split('\\n', 1)[-1]).get('token') if get_status_code(r) == 200 else None; "
        "r2 = creds.request('POST', url, params=params, headers={'x-goog-colab-token': tok}, files={'file_id': (None, 'empty.ipynb')}); "
        "data = json.loads(r2.text.split('\\n', 1)[-1]); "
        "print('READY' if data.get('success') else 'PENDING')"
    )
    proc = subprocess.run([COLAB_PYTHON, "-c", probe], capture_output=True, text=True, env=env)
    return proc.returncode == 0 and "READY" in proc.stdout


def run_drivemount(session: str, env: dict, timeout_sec: float = 90.0) -> bool:
    """Attempt colab drivemount; return True if mounted, False if interactive auth wasn't approved."""
    print(f"[runner] $ colab drivemount -s {session}", flush=True)
    proc = subprocess.Popen(
        ["colab", "drivemount", "-s", session],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    def _stream_output() -> None:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 1024)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

    reader = threading.Thread(target=_stream_output, daemon=True)
    reader.start()

    deadline = time.monotonic() + timeout_sec
    sent_enter = False
    while proc.poll() is None:
        time.sleep(3.0)
        if proc.poll() is not None:
            break
        if not sent_enter and check_drive_auth_ready(session, env):
            print("\n[runner] Drive OAuth confirmed ready; completing drivemount...", flush=True)
            if proc.stdin:
                proc.stdin.write(b"\n")
                proc.stdin.flush()
            sent_enter = True
            proc.wait(timeout=60)
            break
        if time.monotonic() > deadline:
            print(
                f"\n[runner] Interactive Drive OAuth not approved within {int(timeout_sec)}s.",
                flush=True,
            )
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            reader.join(timeout=1.0)
            return False

    reader.join(timeout=2.0)
    return proc.returncode == 0


def provision_gpu_session(session: str, preferred_gpu: str, env: dict) -> str:
    """Allocate a Colab session, trying preferred_gpu first and falling back cleanly."""
    if preferred_gpu.lower() == "none":
        run_cmd(["colab", "new", "-s", session], env=env)
        return "CPU"

    candidates = [preferred_gpu]
    if preferred_gpu.upper() == "T4":
        candidates.append("L4")
    elif preferred_gpu.upper() == "L4":
        candidates.append("T4")

    last_err = None
    for gpu in candidates:
        try:
            run_cmd(["colab", "new", "-s", session, "--gpu", gpu], env=env)
            return gpu
        except RuntimeError as exc:
            last_err = exc
            print(f"[runner] GPU '{gpu}' unavailable ({exc}); trying fallback if any...", flush=True)
    raise RuntimeError(f"Failed to allocate GPU session: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Python script(s) on Colab GPU with Drive mount support and guaranteed teardown."
    )
    parser.add_argument("--script", action="append", required=True, help="Local Python script(s) to execute on Colab.")
    parser.add_argument("--session", default="ml-final", help="Colab session name.")
    parser.add_argument("--gpu", default="T4", help="Preferred GPU type (T4, L4, A100, etc.) or 'none' for CPU.")
    parser.add_argument("--timeout", type=int, default=600, help="Colab exec idle timeout in seconds.")
    parser.add_argument("--try-drivemount", action="store_true", help="Attempt colab drivemount before running scripts.")
    parser.add_argument("--require-drive", action="store_true", help="Fail if Google Drive mount does not succeed.")
    parser.add_argument("--drive-timeout", type=float, default=90.0, help="Drive OAuth wait timeout in seconds.")
    parser.add_argument("--upload", action="append", default=[], help="Upload mapping local:remote.")
    parser.add_argument("--download", action="append", default=[], help="Download mapping remote:local.")
    parser.add_argument("--keep", action="store_true", help="Keep session alive instead of stopping on exit.")
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("REQUEST_TIMEOUT", "60")
    env.setdefault("COLAB_CLI_ASSIGN_RETRIES", "0")

    session = args.session
    created = False
    try:
        gpu_used = provision_gpu_session(session, args.gpu, env=env)
        created = True
        print(f"[runner] Active hardware: {gpu_used}", flush=True)

        run_cmd(
            ["colab", "exec", "-s", session, "--timeout", "60"],
            env=env,
            input_text="import os; print('kernel ready')\n",
        )

        if args.try_drivemount or args.require_drive:
            mounted = run_drivemount(session, env=env, timeout_sec=args.drive_timeout)
            if not mounted:
                print("[runner] Restarting remote kernel to clear pending drivemount cell...", flush=True)
                run_cmd(["colab", "restart-kernel", "-s", session], env=env)
                if args.require_drive:
                    raise RuntimeError("Google Drive mount failed or timed out (--require-drive set).")

        for item in args.upload:
            local_p, remote_p = item.split(":", 1)
            run_cmd(["colab", "upload", "-s", session, local_p, remote_p], env=env)

        for script_path in args.script:
            run_cmd(["colab", "exec", "-s", session, "--timeout", str(args.timeout), "-f", script_path], env=env)

        for item in args.download:
            remote_p, local_p = item.split(":", 1)
            run_cmd(["colab", "download", "-s", session, remote_p, local_p], env=env)

        return 0
    finally:
        if created and not args.keep:
            print(f"[runner] Cleaning up session '{session}'...", flush=True)
            subprocess.run(["colab", "stop", "-s", session], check=False, env=env)
        subprocess.run(["colab", "sessions"], check=False, env=env)


if __name__ == "__main__":
    sys.exit(main())
