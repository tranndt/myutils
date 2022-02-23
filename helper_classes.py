import time
import pandas as pd
from .main import *

# -----------------------------------------------
#   TIMER FOR PROCESSES/TASKS
# -----------------------------------------------

class ProcessTimer:
    def __init__(self) -> None:
        self.start_ = {}
        self.prev_ = {}
        self.curr_ = {}
        self.NEW_ID = 0;

    def start(self,job_id=None):
        job_id = self.NEW_ID if job_id is None else job_id
        self.start_[job_id] = time.time()
        self.curr_[job_id] = self.start_[job_id]
        self.prev_[job_id] = -1
        self.NEW_ID += 1

    def record(self,job_id=0):
        self.prev_[job_id] = self.curr_[job_id]
        self.curr_[job_id] = time.time()

    def execute(self,func,job_id=-1,return_val=True,**func_args):
        """
        Record the time for executing a function.

        Return 
        ---------
        Return the time of execution followed by the function followed by the return value(s)
        """
        self.start(job_id)
        val = func(**func_args)
        self.record(job_id)
        if return_val:
            return self.time_elapsed(job_id),val
        else:
            return self.time_elapsed(job_id)

    def step_elapsed(self,job_id=0):
        return -1 if (job_id not in self.prev_.keys() or job_id not in self.curr_.keys()) else self.curr_[job_id] - self.prev_[job_id]

    def time_elapsed(self,job_id=0):
        return -1 if (job_id not in self.start_.keys() or job_id not in self.curr_.keys()) else self.curr_[job_id] - self.start_[job_id]
        


