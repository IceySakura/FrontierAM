"""IHVP sequence protocol and shared local-development scheduler.

Only current RHS and operator actions cross the worker boundary. This local
process harness is not an adversarial filesystem/network sandbox.
"""
from __future__ import annotations
import json
import os
import select
import struct
import time
from pathlib import Path
import numpy as np

PROTOCOL_VERSION = "ihvp-sequence-v1/json-binary-1"
MAX_HEADER_BYTES = 65536
MAX_PAYLOAD_BYTES = 32*1024**2
MAX_BLOCK_COLUMNS = 64

class BudgetExceeded(RuntimeError):
    pass

class ProtocolError(RuntimeError):
    pass

class InvalidPrediction(ValueError):
    pass


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str,int,float,bool)):
        return value
    return str(value)


def send_message(stream, kind, array=None, **fields):
    fields = dict(fields, kind=kind, protocol=PROTOCOL_VERSION)
    payload = b""
    if array is not None:
        array = np.ascontiguousarray(array, dtype="<f8")
        if array.ndim not in (1,2):
            raise ProtocolError("array must have one or two dimensions")
        payload = array.tobytes(order="C")
        fields.update(shape=list(array.shape), payload_bytes=len(payload))
    else:
        fields["payload_bytes"] = 0
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError("payload too large")
    header = json.dumps(clean(fields), separators=(",",":"), allow_nan=False).encode()
    if len(header) > MAX_HEADER_BYTES:
        raise ProtocolError("header too large")
    for block in (struct.pack("!I",len(header)),header,payload):
        view=memoryview(block)
        while view:
            written=os.write(stream.fileno(),view)
            if written<=0:raise BrokenPipeError("protocol write failed")
            view=view[written:]
    stream.flush()


def _read_exact(stream, length, deadline=None):
    fd = stream.fileno()
    chunks = []
    remaining = length
    while remaining:
        if deadline is not None:
            left = deadline-time.monotonic()
            if left <= 0:
                raise BudgetExceeded("sequence hard deadline")
            if not select.select([fd],[],[],min(left,.05))[0]:
                continue
        block = os.read(fd,min(remaining,65536))
        if not block:
            raise EOFError("worker stream closed")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def receive_message(stream, deadline=None, expected_dimension=None):
    size = struct.unpack("!I",_read_exact(stream,4,deadline))[0]
    if not 0 < size <= MAX_HEADER_BYTES:
        raise ProtocolError("invalid header length")
    header = json.loads(_read_exact(stream,size,deadline))
    if not isinstance(header,dict) or header.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError("protocol version mismatch")
    count = header.get("payload_bytes")
    if type(count) is not int or not 0 <= count <= MAX_PAYLOAD_BYTES:
        raise ProtocolError("invalid payload size")
    array = None
    if count:
        shape = header.get("shape")
        if (not isinstance(shape,list) or len(shape) not in (1,2)
                or any(type(v) is not int or v <= 0 for v in shape)
                or int(np.prod(shape,dtype=object))*8 != count):
            raise ProtocolError("invalid payload shape")
        if expected_dimension is not None and (shape[0] != expected_dimension
                or (len(shape) == 2 and shape[1] > MAX_BLOCK_COLUMNS)):
            raise ProtocolError("array exceeds the current dimension/block contract")
        array = np.frombuffer(_read_exact(stream,count,deadline),dtype="<f8").reshape(shape).copy()
    return header,array


def run_case(submission, case, *, wall_seconds=30., memory_mib=256.,
             seed=1, options=None, target_rtol=1e-8, log_path=None):
    """Shared local scheduler. `case` stays in the host and supplies rhs/hvp/score.

    The oracle deadline is sequence-wide. Per-step allocation is a solver choice.
    Startup/import, IPC, host HVP work, solver work and output consume the clock.
    """
    import subprocess
    import sys
    import threading
    import psutil
    submission = Path(submission).resolve()
    if not (submission/"solver.py").is_file():
        raise FileNotFoundError("submission must contain solver.py")
    d,T = int(case.dimension),int(case.steps)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    argv = [sys.executable,str(Path(__file__).with_name("worker.py")),
            "--submission",str(submission),"--dimension",str(d),"--steps",str(T),
            "--seed",str(seed),"--options",json.dumps(options or {})]
    predictions = np.zeros((T,d),dtype=np.float64)
    records = [{"step":t,"completed":False,"error":None,"hvp_vectors":0,
                "hvp_requests":0,"oracle_seconds":0.} for t in range(T)]
    cap = int(memory_mib*1024**2)
    observed = {"peak":0,"reason":None}
    done_usage = {}
    stop = threading.Event()
    log = open(log_path,"wb") if log_path else subprocess.DEVNULL
    start = time.monotonic()
    hard = start+wall_seconds-.3
    soft = max(start,hard-min(1.2,wall_seconds*.1))
    # FrontierAM grading container: the evaluator runs as root next to root-only teacher data and starts the
    # worker as this unprivileged UID, so the submission can reach only the current RHS and HVP replies.
    drop = {}
    if os.environ.get("IHVP_WORKER_UID") and os.geteuid() == 0:
        uid = int(os.environ["IHVP_WORKER_UID"])
        drop = {"user":uid,"group":uid,"extra_groups":[]}
        env.pop("IHVP_WORKER_UID",None)
        env["HOME"] = str(submission)
    process = subprocess.Popen(argv,cwd=submission,env=env,stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,stderr=log,bufsize=0,**drop)
    child = psutil.Process(process.pid)
    def monitor():
        while not stop.is_set():
            try:
                rss = child.memory_info().rss
                highwater=rss
                if sys.platform.startswith("linux"):
                    highwater=next(int(line.split()[1])*1024 for line in Path(f'/proc/{process.pid}/status').read_text().splitlines() if line.startswith('VmHWM:'))
                observed["peak"] = max(observed["peak"],rss,highwater)
                if observed["peak"] > cap:
                    observed["reason"] = "memory_limit"
                    process.terminate()
                    return
                if time.monotonic() >= hard:
                    observed["reason"] = "wall_limit"
                    process.terminate()
                    return
            except (psutil.NoSuchProcess,ProcessLookupError,FileNotFoundError,StopIteration):
                return
            stop.wait(.005)
    watcher = threading.Thread(target=monitor,daemon=True)
    watcher.start()
    current = -1
    terminal = None
    try:
        while True:
            header,array = receive_message(process.stdout,hard,expected_dimension=d)
            kind = header.get("kind")
            if kind == "ready":
                continue
            if kind == "next":
                tick = header.get("step")
                if type(tick) is not int or tick != current+1 or not 0 <= tick < T:
                    raise ProtocolError("out-of-order step")
                if current >= 0 and not records[current]["completed"]:
                    raise ProtocolError("previous step has no output")
                current = tick
                phase = int(case.phase(tick))
                context = {"deadline":soft,"sequence_deadline":soft,"hard_deadline":hard,
                           "step":tick,"steps":T,"phase":phase,"mu":float(case.mu),
                           "L":float(case.L),"target_rtol":target_rtol,
                           "phase_changed":bool(tick==0 or phase!=int(case.phase(tick-1)))}
                send_message(process.stdin,"step",case.rhs(tick),context=context)
            elif kind == "hvp":
                if header.get("step") != current or current < 0 or records[current]["completed"]:
                    raise ProtocolError("HVP for unavailable system")
                if (array is None or array.shape[0]!=d or not np.isfinite(array).all()
                        or (array.ndim==2 and array.shape[1]>MAX_BLOCK_COLUMNS)):
                    raise ProtocolError("invalid HVP vector/block")
                if time.monotonic() >= soft:
                    send_message(process.stdin,"budget")
                    continue
                begin = time.monotonic()
                with np.errstate(over="ignore",invalid="ignore"):
                    product = np.asarray(case.hvp(current,array),dtype=np.float64)
                if product.shape!=array.shape or not np.isfinite(product).all():
                    raise ProtocolError("current HVP request produced nonfinite numeric values")
                records[current]["oracle_seconds"] += time.monotonic()-begin
                records[current]["hvp_vectors"] += 1 if array.ndim==1 else array.shape[1]
                records[current]["hvp_requests"] += 1
                send_message(process.stdin,"product",product)
            elif kind == "result":
                if header.get("step") != current or current<0 or records[current]["completed"]:
                    raise ProtocolError("output for unavailable system")
                if array is None or array.shape!=(d,) or not np.isfinite(array).all():
                    raise ProtocolError("output must be a finite d-vector")
                predictions[current] = array
                records[current].update(completed=True,error=header.get("error"),
                                        diagnostics=header.get("diagnostics",{}))
            elif kind == "done":
                if current != T-1 or not all(row["completed"] for row in records):
                    raise ProtocolError("incomplete sequence")
                done_usage = header.get("usage",{})
                observed["peak"] = max(observed["peak"],int(done_usage.get("peak_rss_bytes",0)))
                break
            elif kind == "fatal":
                terminal = "submission_exception: "+str(header.get("error"))
                break
            else:
                raise ProtocolError("unknown worker message")
    except BudgetExceeded:
        terminal = "wall_limit"
    except (EOFError,BrokenPipeError):
        terminal = observed["reason"] or "unexpected_worker_exit"
    except Exception as error:
        terminal = f"protocol_failure: {type(error).__name__}: {error}"
    finally:
        if observed["reason"]:
            terminal = observed["reason"]
        if process.poll() is None:
            if terminal:
                process.terminate()
            try:
                process.wait(timeout=max(0.01,min(.15,start+wall_seconds-time.monotonic())))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=.1)
        stop.set();watcher.join(timeout=.02)
        process.stdin.close();process.stdout.close()
        if log_path:
            log.close()
    wall = time.monotonic()-start
    if process.returncode != 0 and terminal is None:
        terminal = "nonzero_worker_exit"
    if observed["peak"]>cap:
        terminal = "memory_limit"
    if wall>wall_seconds:
        terminal = "wall_limit"
    errors = [row["error"] for row in records if row["error"]]
    valid = terminal is None and not errors and all(row["completed"] for row in records)
    # Private truth scoring and reference work are explicitly outside the clock.
    for t,row in enumerate(records):
        try:
            with np.errstate(over="raise",invalid="raise",divide="raise"):
                metrics = case.score(t,predictions[t])
            if not all(np.isfinite(float(metrics[key])) for key in ("energy_loss","relative_residual")):
                raise InvalidPrediction("nonfinite prediction-derived score")
            row.update(clean(metrics))
        except (InvalidPrediction,FloatingPointError,OverflowError) as error:
            valid=False
            terminal=terminal or "invalid_prediction"
            row.update(energy_loss=None,relative_residual=None,numerical_failure=str(error))
    losses = [row["energy_loss"] for row in records]
    residuals = [row["relative_residual"] for row in records]
    finite_losses=[float(value) for value in losses if value is not None]
    return {"protocol":PROTOCOL_VERSION,"valid":valid,"resource_compliant":valid,
            "termination_reason":terminal,"solver_errors":errors,"dimension":d,"steps":T,
            "wall_seconds":wall,"peak_rss_mib":observed["peak"]/1024**2,
            "mean_energy_loss":float(np.mean(finite_losses)) if len(finite_losses)==T else None,
            "max_energy_loss":max(finite_losses) if len(finite_losses)==T else None,
            "fraction_energy_loss_le_1e_8":sum(value is not None and value<=1e-8 for value in losses)/T,
            "fraction_residual_target_met":sum(value is not None and value<=target_rtol*1.05 for value in residuals)/T,
            "completed_steps":sum(row["completed"] for row in records),
            "hvp_calls":sum(row["hvp_vectors"] for row in records),
            "mean_time_limit_seconds":wall_seconds,"sequence_hard_limit_seconds":wall_seconds,
            "memory_limit_mib":memory_mib,"seed":seed,"records":records,
            "memory_meter":"linux_proc_VmHWM_and_sampled_RSS" if sys.platform.startswith("linux") else "rusage_and_sampled_RSS",
            "local_isolation":"protocol separation, not adversarial host filesystem isolation",
            "prediction_array":predictions}
