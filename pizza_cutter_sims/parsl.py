import logging
import os
import re
import time
import typeguard
import atexit

import parsl
from parsl.channels import LocalChannel
from parsl.providers.provider_base import JobState, JobStatus
from parsl.utils import RepresentationMixin
from parsl.launchers import SingleNodeLauncher
from parsl.providers.condor.template import template_string
from parsl.providers.cluster_provider import ClusterProvider
from parsl.providers.error import ScaleOutFailed

from typing import Dict, List, Optional
from parsl.channels.base import Channel
from parsl.launchers.launchers import Launcher

from parsl.config import Config
from parsl.executors import HighThroughputExecutor

from esutil.pbar import PBar

logger = logging.getLogger(__name__)

# See http://pages.cs.wisc.edu/~adesmet/status.html
translate_table = {
    '1': JobState.PENDING,
    '2': JobState.RUNNING,
    '3': JobState.CANCELLED,
    '4': JobState.COMPLETED,
    '5': JobState.FAILED,
    '6': JobState.FAILED,
}


class CondorProvider(RepresentationMixin, ClusterProvider):
    """HTCondor Execution Provider.

    Parameters
    ----------
    channel : Channel
        Channel for accessing this provider. Possible channels include
        :class:`~parsl.channels.LocalChannel` (the default),
        :class:`~parsl.channels.SSHChannel`, or
        :class:`~parsl.channels.SSHInteractiveLoginChannel`.
    nodes_per_block : int
        Nodes to provision per block.
    cores_per_slot : int
        Specify the number of cores to provision per slot. If set to None, executors
        will assume all cores on the node are available for computation. Default is
        None.
    mem_per_slot : float
        Specify the real memory to provision per slot in GB. If set to None, no
        explicit request to the scheduler will be made. Default is None.
    init_blocks : int
        Number of blocks to provision at time of initialization
    min_blocks : int
        Minimum number of blocks to maintain
    max_blocks : int
        Maximum number of blocks to maintain.
    parallelism : float
        Ratio of provisioned task slots to active tasks. A parallelism value of 1
        represents aggressive
        scaling where as many resources as possible are used; parallelism close to 0
        represents
        the opposite situation in which as few resources as possible (i.e., min_blocks)
        are used.
    environment : dict of str
        A dictionary of environmant variable name and value pairs which will be set
        before running a task.
    project : str
        Project which the job will be charged against
    scheduler_options : str
        String to add specific condor attributes to the HTCondor submit script.
    transfer_input_files : list(str)
        List of strings of paths to additional files or directories to transfer
        to the job
    worker_init : str
        Command to be run before starting a worker.
    requirements : str
        Condor requirements.
    launcher : Launcher
        Launcher for this provider. Possible launchers include
        :class:`~parsl.launchers.SingleNodeLauncher` (the default),
    cmd_timeout : int
        Timeout for commands made to the scheduler in seconds.
    cmd_chunk_size : int
        Calls to the scheduler will be made for chunks of blocks with this size.
    """
    @typeguard.typechecked
    def __init__(self,
                 channel: Channel = LocalChannel(),
                 nodes_per_block: int = 1,
                 cores_per_slot: Optional[int] = None,
                 mem_per_slot: Optional[float] = None,
                 init_blocks: int = 1,
                 min_blocks: int = 0,
                 max_blocks: int = 1,
                 parallelism: float = 1,
                 environment: Optional[Dict[str, str]] = None,
                 project: str = '',
                 scheduler_options: str = '',
                 transfer_input_files: List[str] = [],
                 walltime: str = "00:10:00",
                 worker_init: str = '',
                 launcher: Launcher = SingleNodeLauncher(),
                 requirements: str = '',
                 cmd_timeout: int = 60,
                 cmd_chunk_size: int = 100) -> None:

        label = 'condor'
        super().__init__(label,
                         channel,
                         nodes_per_block,
                         init_blocks,
                         min_blocks,
                         max_blocks,
                         parallelism,
                         walltime,
                         launcher,
                         cmd_timeout=cmd_timeout)
        self.cores_per_slot = cores_per_slot
        self.mem_per_slot = mem_per_slot
        self.cmd_chunk_size = cmd_chunk_size

        # To Parsl, Condor slots should be treated equivalently to nodes
        self.cores_per_node = cores_per_slot
        self.mem_per_node = mem_per_slot

        self.environment = environment if environment is not None else {}
        for key, value in self.environment.items():
            # To escape literal quote marks, double them
            # See: http://research.cs.wisc.edu/htcondor/manual/v8.6/condor_submit.html
            try:
                self.environment[key] = "'{}'".format(
                    value.replace("'", '"').replace('"', '""')
                )
            except AttributeError:
                pass

        self.project = project
        self.scheduler_options = scheduler_options + '\n'
        self.worker_init = worker_init + '\n'
        self.requirements = requirements
        self.transfer_input_files = transfer_input_files

        atexit.register(_kill_all_jobs, self)

    def _status(self):
        """Update the resource dictionary with job statuses."""

        for job_id_chunk in _chunker(self.resources.keys(), self.cmd_chunk_size):
            job_id_list = ' '.join(job_id_chunk)
            cmd = "condor_q {0} -af:jr JobStatus".format(job_id_list)
            retcode, stdout, stderr = self.execute_wait(cmd)
            """
            Example output:
            $ condor_q 34524642.0 34524643.0 -af:jr JobStatus
            34524642.0 2
            34524643.0 1
            """

            for line in stdout.splitlines():
                parts = line.strip().split()
                job_id = parts[0]
                state = translate_table.get(parts[1], JobState.UNKNOWN)
                if job_id in self.resources:
                    self.resources[job_id]['status'] = JobStatus(state)

    def status(self, job_ids):
        """Get the status of a list of jobs identified by their ids.
        Parameters
        ----------
        job_ids : list of int
            Identifiers of jobs for which the status will be returned.
        Returns
        -------
        List of int
            Status codes for the requested jobs.
        """
        self._status()
        return [self.resources[jid]['status'] for jid in job_ids]

    def submit(self, command, tasks_per_node, job_name="parsl.condor"):
        """Submits the command onto an Local Resource Manager job.
        example file with the complex case of multiple submits per job:

            Universe =vanilla
            output = out.$(Cluster).$(Process)
            error = err.$(Cluster).$(Process)
            log = log.$(Cluster)
            leave_in_queue = true
            executable = test.sh
            queue 5
            executable = foo
            queue 1

        $ condor_submit test.sub
        Submitting job(s)......
        5 job(s) submitted to cluster 118907.
        1 job(s) submitted to cluster 118908.

        Parameters
        ----------
        command : str
            Command to execute
        job_name : str
            Job name prefix.
        tasks_per_node : int
            command invocations to be launched per node

        Returns
        -------
        None or str
            None if at capacity and cannot provision more; otherwise the
            identifier for the job.
        """

        logger.debug("Attempting to launch")

        job_name = "parsl.{0}.{1}".format(job_name, time.time())

        scheduler_options = self.scheduler_options
        worker_init = self.worker_init
        if self.mem_per_slot is not None:
            scheduler_options += 'RequestMemory = {}\n'.format(self.mem_per_slot * 1024)
            worker_init += 'export PARSL_MEMORY_GB={}\n'.format(self.mem_per_slot)
        if self.cores_per_slot is not None:
            scheduler_options += 'RequestCpus = {}\n'.format(self.cores_per_slot)
            worker_init += 'export PARSL_CORES={}\n'.format(self.cores_per_slot)

        script_path = "{0}/{1}.submit".format(self.script_dir, job_name)
        script_path = os.path.abspath(script_path)
        userscript_path = "{0}/{1}.script".format(self.script_dir, job_name)
        userscript_path = os.path.abspath(userscript_path)

        self.environment["JOBNAME"] = "'{}'".format(job_name)

        job_config = {}
        job_config["job_name"] = job_name
        job_config["submit_script_dir"] = self.channel.script_dir
        job_config["project"] = self.project
        job_config["nodes"] = self.nodes_per_block
        job_config["scheduler_options"] = scheduler_options
        job_config["worker_init"] = worker_init
        job_config["user_script"] = command
        job_config["tasks_per_node"] = tasks_per_node
        job_config["requirements"] = self.requirements
        job_config["environment"] = ' '.join(
            ['{}={}'.format(key, value) for key, value in self.environment.items()]
        )

        # Move the user script
        # This is where the command should be wrapped by the launchers.
        wrapped_command = self.launcher(command,
                                        tasks_per_node,
                                        self.nodes_per_block)

        with open(userscript_path, 'w') as f:
            f.write(job_config["worker_init"] + '\n' + wrapped_command)

        user_script_path = self.channel.push_file(
            userscript_path, self.channel.script_dir
        )
        the_input_files = [user_script_path] + self.transfer_input_files
        job_config["input_files"] = ','.join(the_input_files)
        job_config["job_script"] = os.path.basename(user_script_path)

        # Construct and move the submit script
        self._write_submit_script(template_string, script_path, job_name, job_config)
        channel_script_path = self.channel.push_file(
            script_path, self.channel.script_dir
        )

        cmd = "condor_submit {0}".format(channel_script_path)
        try:
            retcode, stdout, stderr = self.execute_wait(cmd)
        except Exception as e:
            raise ScaleOutFailed(self.label, str(e))

        job_id = []

        if retcode == 0:
            for line in stdout.split('\n'):
                if re.match('^[0-9]', line) is not None:
                    cluster = line.split(" ")[5]
                    # We know the first job id ("process" in condor terms) within a
                    # cluster is 0 and we know the total number of jobs from
                    # condor_submit, so we use some list comprehensions to expand
                    # the condor_submit output into job IDs
                    # e.g., ['118907.0', '118907.1', '118907.2', '118907.3',
                    #        '118907.4', '118908.0']
                    processes = [str(x) for x in range(0, int(line[0]))]
                    job_id += [cluster + process for process in processes]

            self._add_resource(job_id)
            return job_id[0]
        else:
            message = "Command '{}' failed with return code {}".format(cmd, retcode)
            message += (
                " and standard output '{}'".format(stdout.strip())
                if stdout is not None
                else ''
            )
            message += (
                " and standard error '{}'".format(stderr.strip())
                if stderr is not None
                else ''
            )
            raise ScaleOutFailed(self.label, message)

    def cancel(self, job_ids):
        """Cancels the jobs specified by a list of job IDs.

        Parameters
        ----------
        job_ids : list of str
            The job IDs to cancel.

        Returns
        -------
        list of bool
            Each entry in the list will be True if the job is cancelled succesfully,
            otherwise False.
        """
        rets = []
        for job_id_chunk in _chunker(job_ids, self.cmd_chunk_size):
            job_id_list = ' '.join(job_id_chunk)
            cmd = "condor_rm {0}; condor_rm -forcex {0}".format(job_id_list)
            logger.debug("Attempting removal of jobs : {0}".format(cmd))
            retcode, stdout, stderr = self.execute_wait(cmd)
            if retcode == 0:
                for jid in job_id_chunk:
                    if jid in self.resources:
                        self.resources[jid]['status'] = JobStatus(JobState.CANCELLED)
                rets.extend([True for i in job_id_chunk])
            else:
                rets.extend([False for i in job_id_chunk])

        return rets

    def _add_resource(self, job_id):
        for jid in job_id:
            self.resources[jid] = {'status': JobStatus(JobState.PENDING), 'size': 1}
        return True

    @property
    def status_polling_interval(self):
        return 60


# see https://stackoverflow.com/a/18243990
def _chunker(seq, size):
    res = []
    for el in seq:
        res.append(el)
        if len(res) == size:
            yield res
            res = []
    if res:
        yield res


def _kill_all_jobs(condor_provider):
    job_ids = list(condor_provider.resources.keys())
    condor_provider.cancel(job_ids)


class ParslCondorPool():
    """A schwimbad pool-type interface to parsl.

    Parameters
    ----------
    conda_env : str, optional
        The conda environmant to activate. Default is "pizza-cutter-sims".
    walltime_hours : int, optional
        The maximum walltime of a condor job. Default is 12.
    parallelism : float, optional
        How quickly to scale up condor jobs. A value of 1 will lanuch one condor
        job for every mapped process. Default is 0.75.
    verbose : int
        If greater than 0, print a progress bar.
    """

    _condor_preamble = """\
Universe = vanilla
Notification = Never
request_memory = 2G
kill_sig = SIGINT
+Experiment = "astro"
"""

    _worker_init = """\
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
"""

    def __init__(
        self,
        conda_env="pizza-cutter-sims",
        walltime_hours=12,
        parallelism=0.75,
        verbose=0,
    ):
        self._conda_env = conda_env
        self._walltime_hours = walltime_hours
        self._parallelism = parallelism
        self._verbose = verbose
        self._active = False
        self.config = Config(
            strategy='htex_auto_scale',
            executors=[
                HighThroughputExecutor(
                    worker_debug=True,
                    max_workers=1,
                    poll_period=10000,
                    provider=CondorProvider(
                        cores_per_slot=1,
                        # mem_per_slot=2, done in scheduler_options
                        nodes_per_block=1,
                        init_blocks=0,
                        parallelism=self._parallelism,
                        max_blocks=10000,  # 10 seconds in milliseconds
                        scheduler_options=self._condor_preamble,
                        worker_init=self._worker_init % self._conda_env,
                        walltime="%d:00:00" % self._walltime_hours,
                        cmd_timeout=300,  # five minutes
                    )
                )
            ],
        )

    def __enter__(self):
        parsl.load(self.config)
        self._axctive = True

    def __exit__(self, exc_type, exc_value, traceback):
        parsl.clear()
        self._axctive = False

    def map(self, func, args):
        """Map a function over a list of arguments."""
        if not self._active:
            raise RuntimeError("You must call 'map' within a context manager!")

        pfunc = parsl.python_app(func)

        futs = [pfunc(arg) for arg in args]

        results = []
        n_tot = len(futs)
        itr = range(n_tot)
        if self._verbose > 0:
            itr = PBar(itr, n_bars=79)
        for i in itr:
            fut = futs[i]
            # this call sometimes helps the result return properly
            fut.done()
            results.append(fut.result())

        return results
