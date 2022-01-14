import os
import uuid
import subprocess
import cloudpickle
import joblib
import atexit
import time
import threading

from concurrent.futures import ThreadPoolExecutor, Future

DEBUG = False

ACTIVE_THREAD_LOCK = threading.RLock()

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


def _get_job_status(cjob):
    status = None
    res = subprocess.run(
        "condor_q %s -af:jr JobStatus" % cjob,
        shell=True,
        capture_output=True,
    )
    if res.returncode == 0:
        for line in res.stdout.decode("utf-8").splitlines():
            line = line.strip().split(" ")
            if line[0] == cjob:
                status = line[1]
                break
    return status


def _get_all_job_statuses_call(cjobs):
    status = {}
    res = subprocess.run(
        "condor_q %s -af:jr JobStatus" % " ".join(cjobs),
        shell=True,
        capture_output=True,
    )
    if res.returncode == 0:
        for line in res.stdout.decode("utf-8").splitlines():
            line = line.strip().split(" ")
            if line[0] in cjobs:
                status[line[0]] = line[1]
    return status


def _get_all_job_statuses(cjobs):
    status = {}
    jobs_to_check = []
    for cjob in cjobs:
        jobs_to_check.append(cjob)
        if len(jobs_to_check) == 100:
            status.update(_get_all_job_statuses_call(jobs_to_check))
            jobs_to_check = []

    if jobs_to_check:
        status.update(_get_all_job_statuses_call(jobs_to_check))

    return status


def _nanny_function(
    exec, nanny_id
):
    while True:
        time.sleep(1)
        print(
            "%d: %d left to do" % (nanny_id, len(exec._nanny_subids[nanny_id]))
        )
        if exec._done and len(exec._nanny_subids[nanny_id]) == 0:
            return

        subids = list(exec._nanny_subids[nanny_id])
        if DEBUG:
            print("getting job statuses", flush=True)
        statuses = _get_all_job_statuses([
            exec._nanny_subids[nanny_id][subid][0]
            for subid in subids
            if exec._nanny_subids[nanny_id][subid][0] is not None
        ])
        # if DEBUG:
        if any(val == "4" for val in statuses.values()):
            print("%d: got job statuses:" % nanny_id, statuses, flush=True)

        if DEBUG:
            print("%d: looping for %d subids" % (nanny_id, len(subids)), flush=True)
        n_submitted = 0
        for subid in subids:
            cjob = exec._nanny_subids[nanny_id][subid][0]
            fut = exec._nanny_subids[nanny_id][subid][1]
            job_data = exec._nanny_subids[nanny_id][subid][2]

            if cjob is None and job_data is not None:
                if DEBUG:
                    print("attempting to submit %s" % subid, flush=True)
                with ACTIVE_THREAD_LOCK:
                    if exec._num_jobs < exec.max_workers:
                        exec._num_jobs += 1
                        submit_job = True
                    else:
                        submit_job = False

                if submit_job:
                    if DEBUG:
                        print("submitting %s" % subid, flush=True)
                    cjob = exec._submit_condor_job(
                        exec, subid, nanny_id, fut, job_data
                    )

                    if cjob is None:
                        if DEBUG:
                            print("could not submit %s" % subid, flush=True)
                        del exec._nanny_subids[nanny_id][subid]
                    else:
                        if DEBUG:
                            print("submitted %s" % subid, flush=True)
                        fut.cjob = cjob
                        exec._nanny_subids[nanny_id][subid] = (cjob, fut, None)

                n_submitted += 1
                if n_submitted >= 100:
                    break

        n_submitted = 0
        for cjob, status_code in statuses.items():
            subid = None
            for _subid in subids:
                if exec._nanny_subids[nanny_id][_subid][0] == cjob:
                    subid = _subid
                    break
            if subid is not None and status_code in ["4", "3", "5", "7"]:
                print("subid|status:", subid, status_code, flush=True)
                outfile = os.path.abspath(
                    os.path.join(exec.execdir, subid, "output.pkl"))
                infile = os.path.abspath(
                    os.path.join(exec.execdir, subid, "input.pkl"))

                del ALL_CONDOR_JOBS[cjob]
                subprocess.run(
                    "condor_rm %s; condor_rm -forcex %s" % (cjob, cjob),
                    shell=True,
                    check=True,
                    capture_output=True,
                )
                if status_code == "4":
                    for _ in range(10):
                        if os.path.exists(outfile):
                            break
                        time.sleep(1)

                if not os.path.exists(outfile):
                    print(
                        "nofile subid|status|file:", subid, status_code, outfile,
                        flush=True,
                    )

                if os.path.exists(outfile):
                    try:
                        res = joblib.load(outfile)
                    except Exception as e:
                        res = e
                elif status_code in ["3", "5", "7"]:
                    res = RuntimeError(
                        "Condor job %s: status %s" % (
                            subid, STATUS_DICT[status_code]
                        )
                    )
                else:
                    res = RuntimeError(
                        "Condor job %s: no status or job output found!" % subid)

                print("subid:", subid, "made res", flush=True)

                subprocess.run(
                    "rm -f %s %s" % (infile, outfile),
                    shell=True,
                    check=True,
                )

                print("subid:", subid, "removed files", flush=True)
                fut = exec._nanny_subids[nanny_id][subid][1]
                if isinstance(res, Exception):
                    fut.set_exception(res)
                else:
                    fut.set_result(res)

                print("subid:", subid, "set future res", flush=True)

                del exec._nanny_subids[nanny_id][subid]
                with ACTIVE_THREAD_LOCK:
                    exec._num_jobs -= 1

                print("subid:", subid, "cleaned up the job", flush=True)

                n_submitted += 1
                if n_submitted >= 100:
                    break

    print("%d: nanny is going on break! - %d" % (
        nanny_id, len(exec._nanny_subids[nanny_id])
        ), flush=True)


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

run-pickled-task $1 ${tmpdir}/$(basename $2) $3 >& ${tmpdir}/$(basename $3)

mv ${tmpdir}/$(basename $3) $3
mv ${tmpdir}/$(basename $2) $2
"""

    def __init__(
        self, max_workers=10000, conda_env="pizza-cutter-sims",
        verbose=None,
    ):
        self.max_workers = max_workers
        self.execid = uuid.uuid4().hex
        self.execdir = "condor-exec/%s" % self.execid
        self.conda_env = conda_env
        self._exec = None
        self._num_nannies = 10

    def __enter__(self):
        os.makedirs(self.execdir, exist_ok=True)
        print(
            "starting condor executor: "
            "exec dir %s - max workers %s" % (
                self.execdir,
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
        self._exec = ThreadPoolExecutor(max_workers=self._num_nannies)
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

    @staticmethod
    def _submit_condor_job(exec, subid, nanny_id, fut, job_data):
        cjob = None

        if fut.set_running_or_notify_cancel():
            infile = os.path.abspath(os.path.join(exec.execdir, subid, "input.pkl"))
            condorfile = os.path.join(exec.execdir, subid, "condor.sub")
            outfile = os.path.abspath(
                os.path.join(exec.execdir, subid, "output.pkl"))
            logfile = os.path.abspath(os.path.join(exec.execdir, subid, "log.oe"))

            os.makedirs(os.path.join(exec.execdir, subid), exist_ok=True)

            ##############################
            # dump the file
            with open(infile, "wb") as fp:
                cloudpickle.dump(job_data, fp)

            ##############################
            # submit the condor job
            with open(condorfile, "w") as fp:
                fp.write(
                    """\
Universe       = vanilla
Notification   = Never
# this executable must have u+x bits
Executable     = %s
request_memory = 2G
kill_sig       = SIGINT
leave_in_queue = TRUE
+Experiment    = "astro"

+job_name = "%s"
Arguments = %s %s %s
Queue
    """ % (
                        os.path.join(exec.execdir, "run.sh"),
                        "job-%s-%s" % (exec.execid, subid),
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

        return cjob

    def submit(self, func, *args, **kwargs):
        subid = uuid.uuid4().hex
        job_data = joblib.delayed(func)(*args, **kwargs)

        fut = Future()
        fut.execid = self.execid
        fut.subid = subid
        self._nanny_subids[self._nanny_ind][subid] = (None, fut, job_data)

        self._nanny_ind += 1
        self._nanny_ind = self._nanny_ind % self._num_nannies

        return fut
