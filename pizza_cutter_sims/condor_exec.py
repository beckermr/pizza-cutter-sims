import os
import time
import uuid
import subprocess
import cloudpickle
import joblib
import atexit
import threading

from concurrent.futures import ThreadPoolExecutor, Future

ACTIVE_THREAD_LOCK = threading.BoundedSemaphore(value=100)

ALL_CONDOR_JOBS = {}

STATUS_DICT = {
    "": "unknown condor failure :(",
    "1": "Idle",
    "2": "Running",
    "3": "Removed",
    "4": "Completed",
    "5": "Held",
    "6": "Transferring Output",
    "7": "Suspended",
}


def _kill_condor_jobs():
    chunksize = 100
    cjobs = []
    for cjob in list(ALL_CONDOR_JOBS):
        cjobs.append(cjob)
        if len(cjobs) == chunksize:
            _cjobs = " ".join(cjobs)
            subprocess.run("condor_rm " + _cjobs, shell=True, capture_output=True)
            subprocess.run(
                "condor_rm -forcex " + _cjobs, shell=True, capture_output=True)
            cjobs = []

    if cjobs:
        _cjobs = " ".join(cjobs)
        subprocess.run("condor_rm " + _cjobs, shell=True, capture_output=True)
        subprocess.run("condor_rm -forcex " + _cjobs, shell=True, capture_output=True)
        cjobs = []


atexit.register(_kill_condor_jobs)


def _submit_and_poll_function(
    execid, execdir, id, poll_interval, max_poll_time, func, args, kwargs
):
    with ACTIVE_THREAD_LOCK:
        infile = os.path.abspath(os.path.join(execdir, id, "input.pkl"))
        condorfile = os.path.join(execdir, id, "condor.sub")
        outfile = os.path.abspath(os.path.join(execdir, id, "output.pkl"))

        sub = subprocess.run(
            "condor_submit %s" % condorfile,
            shell=True,
            check=True,
            capture_output=True,
        )

        cjob = None
        for line in sub.stdout.decode("utf-8").splitlines():
            line = line.strip()
            if "submitted to cluster" in line:
                line = line.split(" ")
                cjob = line[5] + "0"
                break

        assert cjob is not None
        ALL_CONDOR_JOBS[cjob] = None

    ##############################
    # poll for it being done
    with ACTIVE_THREAD_LOCK:
        check_file = outfile + ".done"
        status_code = None
        timed_out = False
        start_poll = time.time()
    while True:
        time.sleep(poll_interval)

        with ACTIVE_THREAD_LOCK:
            if os.path.exists(check_file):
                break

            res = subprocess.run(
                "condor_q %s -af JobStatus" % cjob,
                shell=True,
                capture_output=True,
            )
            status_code = res.stdout.decode("utf-8").strip()
            if status_code in ["3", "5", "7"]:
                break

            if time.time() - start_poll > max_poll_time:
                timed_out = True
                break

    with ACTIVE_THREAD_LOCK:
        del ALL_CONDOR_JOBS[cjob]
        if os.path.exists(outfile):
            res = joblib.load(outfile)
        elif status_code in ["3", "5", "7"]:
            res = RuntimeError(
                "Condor job %s: status %s" % (id, STATUS_DICT[status_code])
            )
        elif timed_out:
            res = RuntimeError("Condor job %s: timed out after %ss!" % (
                id, max_poll_time)
            )
        else:
            res = RuntimeError("Condor job %s: no status or job output found!" % id)

        subprocess.run(
            "rm -f %s %s %s.done" % (infile, outfile, outfile),
            shell=True,
            check=True,
        )

        if isinstance(res, Exception):
            raise res
        else:
            return res


def _nanny_function(
    exec, nanny_id
):
    while True:
        if exec._done and len(exec._nanny_subids[nanny_id]) == 0:
            return

        subids = list(exec._nanny_subids[nanny_id])
        for subid in subids:
            infile = os.path.abspath(os.path.join(exec.execdir, subid, "input.pkl"))
            outfile = os.path.abspath(os.path.join(exec.execdir, subid, "output.pkl"))
            check_file = outfile + ".done"
            cjob = exec._nanny_subids[nanny_id][subid][0]
            done = False

            if os.path.exists(check_file):
                done = True

            if not done:
                res = subprocess.run(
                    "condor_q %s -af JobStatus" % cjob,
                    shell=True,
                    capture_output=True,
                )
                status_code = res.stdout.decode("utf-8").strip()

                if status_code in ["3", "5", "7"]:
                    done = True

            if done:
                del ALL_CONDOR_JOBS[cjob]
                if os.path.exists(outfile):
                    res = joblib.load(outfile)
                elif status_code in ["3", "5", "7"]:
                    res = RuntimeError(
                        "Condor job %s: status %s" % (subid, STATUS_DICT[status_code])
                    )
                    res = RuntimeError(
                        "Condor job %s: no status or job output found!" % subid)

                subprocess.run(
                    "rm -f %s %s %s.done" % (infile, outfile, outfile),
                    shell=True,
                    check=True,
                )

                fut = exec._nanny_subids[nanny_id][subid][1]
                if isinstance(res, Exception):
                    fut.set_exception(res)
                else:
                    fut.set_result(res)

                del exec._nanny_subids[nanny_id][subid]
                exec._num_jobs -= 1


class CondorExecutor():
    _worker_init = """\
#!/bin/bash

source ~/.bashrc

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    # the condor system creates a scratch directory for us,
    # and cleans up afterward
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    # otherwise use the TMPDIR
    tmpdir='.'
    mkdir -p $tmpdir
fi

source activate %s

run-pickled-task $1 $2 $3 >& ${tmpdir}/$(basename $3)

mv ${tmpdir}/$(basename $3) $3
"""

    def __init__(
        self, max_workers=10000, poll_interval=60, conda_env="pizza-cutter-sims",
        verbose=None, job_timeout=7200,
    ):
        self.max_workers = max_workers
        self.execid = uuid.uuid4().hex
        self.execdir = "condor-exec/%s" % self.execid
        self.conda_env = conda_env
        self._exec = None
        if poll_interval is None:
            self.poll_interval = max(
                (600-10)/(10000 - 100) * (self.max_workers - 100) + 10,
                10,
            )
        else:
            self.poll_interval = poll_interval
        self.job_timeout = job_timeout
        self._num_nannies = 1

    def __enter__(self):
        os.makedirs(self.execdir, exist_ok=True)
        print(
            "starting condor executor: "
            "exec dir %s - poll interval %ss - max workers %s" % (
                self.execdir,
                self.poll_interval,
                self.max_workers,
            ),
            flush=True,
        )

        with open(os.path.join(self.execdir, "run.sh"), "w") as fp:
            fp.write(self._worker_init % self.conda_env)
        subprocess.run(
            "chmod u+x " + os.path.join(self.execdir, "run.sh"),
            shell=True,
            check=True,
        )
        self._exec = ThreadPoolExecutor(max_workers=1)
        self._done = False
        self._nanny_subids = [{} for _ in range(self._num_nannies)]
        self._num_jobs = 0
        self._nanny_ind = 0
        self._nanny_futs = [
            self._exec.submit(_nanny_function, self, i)
            for i in range(self._num_nannies)
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._done = True
        self._exec.shutdown()
        self._exec = None

    def submit(self, func, *args, **kwargs):
        subid = uuid.uuid4().hex

        infile = os.path.abspath(os.path.join(self.execdir, subid, "input.pkl"))
        condorfile = os.path.join(self.execdir, subid, "condor.sub")
        outfile = os.path.abspath(os.path.join(self.execdir, subid, "output.pkl"))
        logfile = os.path.abspath(os.path.join(self.execdir, subid, "log.oe"))

        os.makedirs(os.path.join(self.execdir, subid), exist_ok=True)

        ##############################
        # dump the file
        with open(infile, "wb") as fp:
            cloudpickle.dump(joblib.delayed(func)(*args, **kwargs), fp)

        ##############################
        # submit the condor job
        with open(condorfile, "w") as fp:
            fp.write(
                """\
Universe = vanilla
Notification = Never
# Run this exe with these args
Executable = %s
# Image_Size =  2500000
request_memory = 2G
kill_sig = SIGINT
+Experiment = "astro"

+job_name = "%s"
Arguments = %s %s %s
Queue
""" % (
                    os.path.join(self.execdir, "run.sh"),
                    "job-%s-%s" % (self.execid, subid),
                    infile,
                    outfile,
                    logfile,
                ),
            )

        sub = subprocess.run(
            "condor_submit %s" % condorfile,
            shell=True,
            check=True,
            capture_output=True,
        )

        cjob = None
        for line in sub.stdout.decode("utf-8").splitlines():
            line = line.strip()
            if "submitted to cluster" in line:
                line = line.split(" ")
                cjob = line[5] + "0"
                break

        assert cjob is not None
        ALL_CONDOR_JOBS[cjob] = None

        fut = Future()
        fut.set_running_or_notify_cancel()
        fut.execid = self.execid
        fut.subid = subid
        self._nanny_subids[self._nanny_ind][subid] = (cjob, fut)

        self._num_jobs += 1

        self._nanny_ind += 1
        self._nanny_ind = self._nanny_ind % self._num_nannies

        return fut
