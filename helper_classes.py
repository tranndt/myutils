from .utils import *
import time

# -----------------------------------------------
#   DICT LIKE DUMMY OBJECT
# -----------------------------------------------
class MyObj():
    """
    @Description: Object that serves as a namespace for related attributes within other classes
    @Example:
        cpa = CPA1()\n
        cpa.var1 = MyObj()\n
        cpa.var1.inputs = [1,2,3]\n
        print(cpa.var1.inputs)\n
        >> [1, 2, 3]\n
        cpa.var2 = MyObj({'a':1,'b':2})\n
        print(cpa.var2.a,cpa.var2.b)\n
        >> 1 2

    """
    def __init__(self,__name__="My Object",**kwargs):
        super(MyObj,self)
        self.__name__ = __name__
        self.dict_ = {}
        self.update(**kwargs)

    def update(self,**kwargs):
        if kwargs:
            self.dict_.update(kwargs)
            for key,value in kwargs.items():
                self.__setattr__(key,value)
    
    def dict(self,keys=None):
        """
        Return the dictionary for all or a subset of attributes
        """
        if isNone(keys):
            return self.dict_
        else:
            return dict_subset(self.dict_,keys)

    def __str__(self):
        return f'{self.__name__} {self.dict_}'


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

    def execute(self,func,job_id=-1,**func_args):
        """
        Record the time for executing a function.

        Return 
        ---------
        Return the time of execution followed by the function followed by the return value(s)
        """
        self.start(job_id)
        return_val = func(**func_args)
        self.record(job_id)
        if isNone(return_val):
            return self.time_elapsed(job_id)
        else:
            return self.time_elapsed(job_id),return_val

    def step_elapsed(self,job_id=0):
        return -1 if (job_id not in self.prev_.keys() or job_id not in self.curr_.keys()) else self.curr_[job_id] - self.prev_[job_id]

    def time_elapsed(self,job_id=0):
        return -1 if (job_id not in self.start_.keys() or job_id not in self.curr_.keys()) else self.curr_[job_id] - self.start_[job_id]
        


