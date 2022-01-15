import os
import uuid
import subprocess
import cloudpickle
import joblib
import atexit
import threading
import logging
import time

from concurrent.futures import ThreadPoolExecutor, Future

LOGGER = logging.getLogger("condor_exec")

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

WORKER_INIT = """\
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

mkdir -p $(dirname $2)
mkdir -p $(dirname $3)

run-pickled-task $1 $2 $3 &> $3
"""


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

    for cjob in list(status):
        if cjob not in cjobs:
            del status[cjob]

    return status


def _submit_condor_job(exec, subid, nanny_id, fut, job_data):
    cjob = None

    if not fut.cancelled():
        infile = os.path.join(exec.execdir, subid, "input.pkl")
        condorfile = os.path.join(exec.execdir, subid, "condor.sub")
        outfile = os.path.join(exec.execdir, subid, "output.pkl")
        logfile = os.path.join(exec.execdir, subid, "log.oe")

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
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
preserve_relative_paths = TRUE
transfer_input_files = %s

+job_name = "%s"
transfer_output_files = %s,%s
Arguments = %s %s %s
Queue
""" % (
                    os.path.join(exec.execdir, "run.sh"),
                    infile,
                    "job-%s-%s" % (exec.execid, subid),
                    outfile,
                    logfile,
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


def _attempt_submit(exec, nanny_id, subid):
    submitted = False
    cjob = exec._nanny_subids[nanny_id][subid][0]
    fut = exec._nanny_subids[nanny_id][subid][1]
    job_data = exec._nanny_subids[nanny_id][subid][2]

    if cjob is None and job_data is not None:
        LOGGER.debug("submitting condor job for subid %s", subid)
        with ACTIVE_THREAD_LOCK:
            if exec._num_jobs < exec.max_workers:
                exec._num_jobs += 1
                submit_job = True
            else:
                submit_job = False

        if submit_job:
            cjob = _submit_condor_job(
                exec, subid, nanny_id, fut, job_data
            )

            if cjob is None:
                LOGGER.debug("could not submit condor job for subid %s", subid)
                del exec._nanny_subids[nanny_id][subid]
            else:
                LOGGER.debug("submitted condor job %s for subid %s", cjob, subid)
                fut.cjob = cjob
                exec._nanny_subids[nanny_id][subid] = (cjob, fut, None)
                submitted = True

    return submitted


def _attempt_result(exec, nanny_id, cjob, subids, status_code):
    didit = False
    subid = None
    for _subid in subids:
        if exec._nanny_subids[nanny_id][_subid][0] == cjob:
            subid = _subid
            break
    if subid is not None and status_code in ["4", "3", "5", "7"]:
        outfile = os.path.join(exec.execdir, subid, "output.pkl")
        infile = os.path.join(exec.execdir, subid, "input.pkl")

        del ALL_CONDOR_JOBS[cjob]
        subprocess.run(
            "condor_rm %s; condor_rm -forcex %s" % (cjob, cjob),
            shell=True,
            capture_output=True,
        )

        if not os.path.exists(outfile):
            LOGGER.debug(
                "output %s does not exist for subid %s, condor job %s",
                outfile,
                subid,
                cjob,
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

        subprocess.run(
            "rm -f %s %s" % (infile, outfile),
            shell=True,
        )

        fut = exec._nanny_subids[nanny_id][subid][1]
        if isinstance(res, Exception):
            fut.set_exception(res)
        else:
            fut.set_result(res)

        exec._nanny_subids[nanny_id][subid] = (None, None, None)
        with ACTIVE_THREAD_LOCK:
            exec._num_jobs -= 1

        didit = True

    return didit


def _nanny_function(
    exec, nanny_id, poll_delay,
):
    LOGGER.info("nanny %d started for exec %s", nanny_id, exec.execid)

    try:
        while True:
            subids = [
                k for k in list(exec._nanny_subids[nanny_id])
                if exec._nanny_subids[nanny_id][k][1] is not None
            ]

            if exec._done and len(subids) == 0:
                break

            if len(subids) > 0:
                n_to_submit = sum(
                    1
                    for subid in subids
                    if (
                        exec._nanny_subids[nanny_id][subid][0] is None
                        and
                        exec._nanny_subids[nanny_id][subid][2] is not None
                    )
                )
                if n_to_submit > 0:
                    n_submitted = 0
                    for subid in subids:
                        if _attempt_submit(exec, nanny_id, subid):
                            n_submitted += 1
                        if n_submitted >= 100:
                            break
                elif poll_delay > 0:
                    time.sleep(poll_delay)

                statuses = _get_all_job_statuses([
                    exec._nanny_subids[nanny_id][subid][0]
                    for subid in subids
                    if exec._nanny_subids[nanny_id][subid][0] is not None
                ])
                n_checked = 0
                for cjob, status_code in statuses.items():
                    if _attempt_result(exec, nanny_id, cjob, subids, status_code):
                        n_checked += 1
                    if n_checked >= 100:
                        break

            elif poll_delay > 0:
                time.sleep(poll_delay)

        subids = [
            k for k in list(exec._nanny_subids[nanny_id])
            if exec._nanny_subids[nanny_id][k][1] is not None
        ]

        LOGGER.info(
            "nanny %d for exec %s is finishing w/ %d subids left",
            nanny_id, exec.execid, subids,
        )
    except Exception as e:
        LOGGER.critical(
            "nanny %d failed! - %s", nanny_id, repr(e)
        )


class CondorExecutor():

    def __init__(
        self, max_workers=10000, conda_env="pizza-cutter-sims",
        verbose=0,
    ):
        self.max_workers = max_workers
        self.execid = uuid.uuid4().hex
        self.execdir = "condor-exec/%s" % self.execid
        self.conda_env = conda_env
        self._exec = None
        self._num_nannies = 10
        self.verbose = verbose

    def __enter__(self):
        os.makedirs(self.execdir, exist_ok=True)
        if self.verbose > 0:
            print(
                "starting condor executor: "
                "exec dir %s - max workers %s" % (
                    self.execdir,
                    self.max_workers,
                ),
                flush=True,
            )

        with open(os.path.join(self.execdir, "run.sh"), "w") as fp:
            fp.write(WORKER_INIT % self.conda_env)
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
            self._exec.submit(
                _nanny_function,
                self,
                i,
                max(1, self._num_nannies/10),
            )
            for i in range(self._num_nannies)
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._done = True
        self._exec.shutdown()
        self._exec = None

    def submit(self, func, *args, **kwargs):
        subid = uuid.uuid4().hex
        job_data = joblib.delayed(func)(*args, **kwargs)

        fut = Future()
        fut.execid = self.execid
        fut.subid = subid
        self._nanny_subids[self._nanny_ind][subid] = (None, fut, job_data)
        fut.set_running_or_notify_cancel()

        self._nanny_ind += 1
        self._nanny_ind = self._nanny_ind % self._num_nannies

        return fut
