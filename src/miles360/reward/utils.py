### This is a collection of utility functions for reward computation maintained by rl360

from miles.rollout.rm_hub.math_utils import _strip_string
import contextlib
import faulthandler
import io
import builtins
import os
import shutil
import sys
import subprocess
import platform
import tempfile
import queue  # Import the queue module for exception type hint
import signal
import multiprocessing
from functools import wraps
from enum import Enum
from pydantic import BaseModel
from typing import Any, Callable, Optional, Dict

import random
import logging
logger = logging.getLogger(__name__)

def pick_judge_url(env_var: str = "IF_LLM_JUDGE_URL") -> str:
    """Return one judge base URL, chosen randomly from a comma-separated env var."""
    raw = os.getenv(env_var, "")
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    if not urls:
        raise EnvironmentError(f"{env_var} is not set")
    return random.choice(urls)


def parse_judge_yes_no(text: str | None) -> bool:
    """Parse a YES/NO judge response robustly.

    Uses rfind so the last 'Final Decision: Yes/No' wins — handles cases where
    the judge echoes both options from the prompt earlier in the response.
    Falls back to the last word being YES/NO if no 'Final Decision:' line is present.
    """
    upper = (text or "").upper().strip()
    yes_pos = upper.rfind("FINAL DECISION: YES")
    no_pos = upper.rfind("FINAL DECISION: NO")
    if yes_pos != -1 or no_pos != -1:
        return yes_pos > no_pos
    words = upper.split()
    return words[-1] == "YES" if words else False


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        if verbose:
            logger.warning("Both strings are None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            logger.info(f"Comparing strings: '{ss1}' and '{ss2}'")
        return ss1 == ss2
    except Exception as e:
        logger.error(f"Error comparing strings: {e}")
        return str1 == str2

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

# --- Top-level helper for multiprocessing timeout ---
# This function MUST be defined at the top level to be pickleable
def _mp_target_wrapper(target_func: Callable, mp_queue: multiprocessing.Queue, args: tuple, kwargs: dict[str, Any]):
    """
    Internal wrapper function executed in the child process.
    Calls the original target function and puts the result or exception into the queue.
    """
    try:
        result = target_func(*args, **kwargs)
        mp_queue.put((True, result))  # Indicate success and put result
    except Exception as e:
        # Ensure the exception is pickleable for the queue
        try:
            import pickle

            pickle.dumps(e)  # Test if the exception is pickleable
            mp_queue.put((False, e))  # Indicate failure and put exception
        except (pickle.PicklingError, TypeError):
            # Fallback if the original exception cannot be pickled
            mp_queue.put((False, RuntimeError(f"Original exception type {type(e).__name__} not pickleable: {e}")))

def timeout_limit(seconds: float, use_signals: bool = False):
    """
    Decorator to add a timeout to a function.

    Args:
        seconds: The timeout duration in seconds.
        use_signals: (Deprecated)  This is deprecated because signals only work reliably in the main thread
                     and can cause issues in multiprocessing or multithreading contexts.
                     Defaults to False, which uses the more robust multiprocessing approach.

    Returns:
        A decorated function with timeout.

    Raises:
        TimeoutError: If the function execution exceeds the specified time.
        RuntimeError: If the child process exits with an error (multiprocessing mode).
        NotImplementedError: If the OS is not POSIX (signals are only supported on POSIX).
    """

    def decorator(func):
        if use_signals:
            if os.name != "posix":
                raise NotImplementedError(f"Unsupported OS: {os.name}")
            # Issue deprecation warning if use_signals is explicitly True
            print(
                "WARN: The 'use_signals=True' option in the timeout decorator is deprecated. \
                Signals are unreliable outside the main thread. \
                Please use the default multiprocessing-based timeout (use_signals=False)."
            )

            @wraps(func)
            def wrapper_signal(*args, **kwargs):
                def handler(signum, frame):
                    # Update function name in error message if needed (optional but good practice)
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (signal)!")

                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                # Use setitimer for float seconds support, alarm only supports integers
                signal.setitimer(signal.ITIMER_REAL, seconds)

                try:
                    result = func(*args, **kwargs)
                finally:
                    # Reset timer and handler
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result

            return wrapper_signal
        else:
            # --- Multiprocessing based timeout (existing logic) ---
            @wraps(func)
            def wrapper_mp(*args, **kwargs):
                q = multiprocessing.Queue(maxsize=1)
                process = multiprocessing.Process(target=_mp_target_wrapper, args=(func, q, args, kwargs))
                process.start()
                process.join(timeout=seconds)

                if process.is_alive():
                    process.terminate()
                    process.join(timeout=0.5)  # Give it a moment to terminate
                    if process.is_alive():
                        try:
                            process.kill()
                            print(f"Warning: Escalated to force kill process {process.pid}")
                        except AttributeError:
                            os.kill(process.pid, signal.SIGKILL)
                            print(f"Fall back to very old way of killing process")
                        process.join(timeout=0.5) 
                    
                    if process.is_alive():
                        print(f"Warning: Process {process.pid} did not terminate gracefully after timeout.")
                        raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds and could not be killed (pid={process.pid})!")

                    # Update function name in error message if needed (optional but good practice)
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (multiprocessing)!")

                try:
                    success, result_or_exc = q.get(timeout=0.1)  # Small timeout for queue read
                    if success:
                        return result_or_exc
                    else:
                        raise result_or_exc  # Reraise exception from child
                except queue.Empty as err:
                    exitcode = process.exitcode
                    if exitcode is not None and exitcode != 0:
                        raise RuntimeError(
                            f"Child process exited with error (exitcode: {exitcode}) before returning result."
                        ) from err
                    else:
                        # Should have timed out if queue is empty after join unless process died unexpectedly
                        # Update function name in error message if needed (optional but good practice)
                        raise TimeoutError(
                            f"Operation timed out or process finished unexpectedly without result "
                            f"(exitcode: {exitcode})."
                        ) from err
                finally:
                    q.close()
                    q.join_thread()

            return wrapper_mp

    return decorator

## Utils for sandboxfusion execution

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 30   # 30 seconds is the default timeout for the executor

class RunStatus(str, Enum):
    # all command finished successfully
    Success = 'Success'
    # one of the process has non-zero return code
    Failed = 'Failed'
    # error on sandbox side
    SandboxError = 'SandboxError'


class CommandRunStatus(str, Enum):
    Finished = 'Finished'
    Error = 'Error'
    TimeLimitExceeded = 'TimeLimitExceeded'


class CommandRunResult(BaseModel):
    status: CommandRunStatus
    execution_time: Optional[float] = None
    return_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

class RunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}

### utils for cruxeval

def check_correctness(check_program, timeout=3):
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.
    """
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(check_program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return result[0] == "passed"

def unsafe_execute(check_program, result, timeout):

    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Run program.
        try:
            @timeout_limit(seconds=timeout)
            def _exec_with_timeout():
                exec_globals = {}
                with swallow_io():
                    exec(check_program, exec_globals)
            
            _exec_with_timeout()
            result.append("passed")
        except TimeoutError:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname

class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"

@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
