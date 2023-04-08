import copy
import numpy as np
import random
import json
from collections import defaultdict

from alns import State

### Parser to parse instance json file ###
# You should not change this class!
class Parser(object):
    
    def __init__(self, json_file):
        '''initialize the parser, saves the data from the file into the following instance variables:
        - 
        Args:
            json_file::str
                the path to the xml file
        '''
        self.json_file = json_file
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.name = self.data['name']
        self.Alpha = self.data['Alpha']
        self.T = self.data['T']
        self.BMAX = self.data['BMax']
        self.WMAX = self.data['WMax']
        self.RMIN = self.data['RMin']

        self.workers = [Worker(worker_data, self.T, self.BMAX, self.WMAX, self.RMIN) for worker_data in self.data['Workers']]
        self.tasks = [Task(task_data) for task_data in self.data['Tasks']]


class Worker(object):
    
    def __init__(self, data, T, bmax, wmax, rmin):
        '''Initialize the worker
        Attributes:
            id::int
                id of the worker
            skills::[skill]
                a list of skills of the worker
            available::{k: v}
                key is the day, value is the list of two elements, 
                the first element in the value is the first available hour for that day,
                the second element in the value is the last available hour for that day, inclusively
            bmax::int
                maximum length constraint
            wmax::int
                maximum working hours
            rmin::int
                minimum rest time
            rate::int
                hourly rate
            tasks_assigned::[task]
                a list of task objects
            blocks::{k: v}
                key is the day where a block is assigned to this worker
                value is the list of two elements
                the first element is the hour of the start of the block
                the second element is the hour of the end of the block, inclusively
                if a worker is not assigned any tasks for the day, the key is removed from the blocks dictionary:
                        Eg. del self.blocks[D]

            total_hours::int
                total working hours for the worker
            
        '''
        self.id = data['w_id']
        self.skills = data['skills']
        self.T = T
        self.available = {int(k):v for k,v in data['available'].items()}
        # the constant number for f2 in the objective function
        self.bmin = 4
        self.bmax = bmax
        self.wmax = wmax
        self.rmin = rmin
        
        self.rate = data['rate']
        self.tasks_assigned = []
        self.blocks = {}
        self.total_hours = 0


    def can_assign(self, task):

        ## check skill set
        if task.skill not in self.skills:
            return False

        ## check available time slots
        if task.day not in self.available.keys():
            return False
        else:
            if task.hour < self.available[task.day][0] or task.hour > self.available[task.day][1]:
                return False

        ## cannot do two tasks at the same time
        for t in self.tasks_assigned:
            if t.day == task.day and t.hour == task.hour:
                return False

        ## If no other tasks assigned in the same day
        if task.day not in self.blocks.keys():
            ## check if task.hour within possible hours for current day
            if task.day-1 in self.blocks.keys() and 23 - self.blocks[task.day-1][1] + task.hour < self.rmin:
                    return False
            if task.day+1 in self.blocks.keys() and 23 - task.hour + self.blocks[task.day+1][0] < self.rmin:
                    return False
            ## check if after total_hours < wmax after adding block
            if self.total_hours + 1 > self.wmax:
                return False

        ## If there are other tasks assigned in the same day
        else:
            ## if the task fits within the existing range
            if task.hour >= self.blocks[task.day][0] and task.hour <= self.blocks[task.day][1]:
                return True

            ## otherwise check if new range after task is assigned is rmin feasible
            else: 
                if task.day-1 in self.blocks.keys() and 23 - self.blocks[task.day-1][1] + task.hour < self.rmin:
                        return False
                if task.day+1 in self.blocks.keys() and 23 - task.hour + self.blocks[task.day+1][0] < self.rmin:
                        return False
        
                ## check if new range after task is assigned is within bmax and wmax
                if task.hour - self.blocks[task.day][0] + 1 > self.bmax or self.blocks[task.day][1] - task.hour + 1 > self.bmax:
                    return False

                if task.hour < self.blocks[task.day][0] and (self.blocks[task.day][0] - task.hour) + self.total_hours > self.wmax: 
                    return False
                if task.hour > self.blocks[task.day][1] and (task.hour - self.blocks[task.day][1] + 1) + self.total_hours > self.wmax: 
                    return False
        return True

    def assign_task(self, task): #takes in the task and assigns the task to the worker
        
        # updates attribute tasks_assigned
        self.tasks_assigned.append(task)

        # updates attribute blocks and total_hours
        if task.day not in self.blocks.keys():
            self.blocks[task.day] = [task.hour, task.hour]
            self.total_hours += 1
        else:
            if task.hour < self.blocks[task.day][0]:
                self.total_hours += self.blocks[task.day][0] - task.hour
                self.blocks[task.day][0] = task.hour 
            elif task.hour > self.blocks[task.day][1]:
                self.total_hours += (task.hour - self.blocks[task.day][1])
                self.blocks[task.day][1] = task.hour
            else:
                # if task.hour >= self.blocks[task.day][0] and task.hour <= self.blocks[task.day][1]:
                # there is no change to self.blocks and self.total_hours
                # self.blocks = self.blocks
                # self.total_hours = self.total_hours
                pass

    def remove_task(self, task_id):
        # find the task_object
        for t in self.tasks_assigned:
            if t.id == task_id:
                task_day = t.day
                task_hour = t.hour
                
                # update tasks_assigned
                self.tasks_assigned.remove(t)

                # list of hours of tasks in the same day
                sameday_hrs = []
                for t1 in self.tasks_assigned:
                    if t1.day == task_day:
                        sameday_hrs.append(t1.hour)
                
                # if no other tasks in the same day
                if len(sameday_hrs) == 0:
                    # update blocks and total_hours
                    self.blocks.pop(task_day)
                    self.total_hours -= 1
                # if there are other tasks in the same day
                else: 
                    if task_hour < min(sameday_hrs):
                        # update blocks and total_hours
                        self.blocks[task_day][0] = min(sameday_hrs)
                        self.total_hours -= min(sameday_hrs) - task_hour
                    elif task_hour > max(sameday_hrs):
                        # update blocks and total_hours
                        self.blocks[task_day][1] = max(sameday_hrs)
                        self.total_hours -=  (task_hour - max(sameday_hrs))
                    else:
                        # min(sameday_hrs) <= task_hour and task_hour <= max(sameday_hrs)
                        pass
                return True
        return False

    def get_objective(self):
        t = sum(max(x[1]-x[0]+1,self.bmin) for x in self.blocks.values())
        return t * self.rate

    def __repr__(self):
        if len(self.blocks) == 0:
            return ''
        return '\n'.join([f'Worker {self.id}: Day {d} Hours {self.blocks[d]} Tasks {sorted([t.id for t in self.tasks_assigned if t.day == d])}' for d in sorted(self.blocks.keys())])


class Task(object):
    
    def __init__(self, data):
        self.id = data['t_id']
        self.skill = data['skill']
        self.day = data['day']
        self.hour = data['hour']


### GSP state class ###
# GSP state class. You could and should add your own helper functions to the class
# But please keep the rest untouched!
class GSP(State):
    
    def __init__(self, name, workers, tasks, alpha):
        '''Initialize the GSP state
        Args:
            name::str
                name of the instance
            workers::[Worker]
                workers of the instance
            tasks::[Task]
                tasks of the instance
        '''
        self.name = name
        self.workers = workers
        self.tasks = tasks
        self.Alpha = alpha
        # the tasks assigned to each worker, eg. [worker1.tasks_assigned, worker2.tasks_assigned, ..., workerN.tasks_assigned]
        self.solution = []
        self.unassigned = list(tasks)
        self.assigned = [] # list of task_ids which were assigned to workers
        self.taskID_taskObj = {t.id : t for t in tasks} # dictionary containing mapping from task_id to its corresponding task object
        self.skill_task_d = defaultdict(list) # skill as key and tasks requiring that skill as value
        self.number_skills = 0 # number of unique skills
    
    def random_initialize(self, seed=None):
        '''
        Args:
            seed::int
                random seed
        Returns:
            objective::float
                objective value of the state
        '''
        if seed is None:
            seed = 606

        random.seed(seed)
        
        
        # Obtain number of skills:
        number_skills_dict={}
        for w in self.workers:
            for skill in w.skills:
                number_skills_dict[skill] = 0
        self.number_skills = len(number_skills_dict.keys())
        del number_skills_dict
        
        def find_best_assignment(self, seed=None, reverse=0):
            '''
            Args:
                seed::int
                    random seed
                reverse:: int
                    to determine how we sort task by skills in self.skill_task_d 
            '''
            if seed is None:
                seed = 606

            random.seed(seed)
            
            # Create a dictionary: skill as key and a list of task requiring that skill as value
            self.skill_task_d = defaultdict(list)
            for t in self.tasks:
                self.skill_task_d[t.skill].append(t) 
             
            # Order skills 
            if reverse == 0:
                self.skill_task_d = {k:v for k,v in sorted(self.skill_task_d.items(), key=lambda x: len(x[1]), reverse=False)}
            else: 
                self.skill_task_d = {k:v for k,v in sorted(self.skill_task_d.items(), key=lambda x: len(x[1]), reverse=True)}

            # select tasks of high important skills first, and low important skills after
            for _, task_list in self.skill_task_d.items():
                random.shuffle(task_list)
                for t in task_list:
                    # Assign t to a worker that incurs minimum increase in cost
                    min_cost_increase = float('inf')
                    assigned_worker = 0
                    for w in self.workers:
                        if w.can_assign(t):
                            # calculate cost increase
                            cur_obj = w.get_objective()
                            w.assign_task(t)
                            updated_obj = w.get_objective()

                            utilization = len(w.tasks_assigned)/(w.blocks[t.day][1]-w.blocks[t.day][0]+1)

                            if t.day - 1 not in w.available and t.day + 1 not in w.available:
                                cost_increase = (updated_obj - cur_obj)/(w.rate*max(w.bmax-4,4))*(len(w.skills)/self.number_skills) + \
                                                (len(w.skills)/self.number_skills)-0.1*utilization
                            elif t.day - 1 in w.available and t.day + 1 not in w.available:
                                cost_increase = (updated_obj - cur_obj)/(w.rate*max(w.bmax-4,4))*(len(w.skills)/self.number_skills) + \
                                                (len(w.skills)/self.number_skills) + \
                                                max(w.available[t.day-1][1] - (23-(w.rmin-t.hour)),0)/24 -0.1*utilization
                            elif t.day - 1 not in w.available and t.day + 1 in w.available:
                                cost_increase = (updated_obj - cur_obj)/(w.rate*max(w.bmax-4,4))*(len(w.skills)/self.number_skills) + \
                                                (len(w.skills)/self.number_skills) + \
                                                max((w.rmin-(23-t.hour))- w.available[t.day+1][0] ,0)/24 -0.1*utilization
                            else:
                                cost_increase = (updated_obj - cur_obj)/(w.rate*max(w.bmax-4,4))*(len(w.skills)/self.number_skills) + \
                                                (len(w.skills)/self.number_skills) + \
                                                max(w.available[t.day-1][1] - (23-(w.rmin-t.hour)),0)/24 + \
                                                max((w.rmin-(23-t.hour))- w.available[t.day+1][0] ,0)/24 -0.1*utilization

                            if cost_increase < min_cost_increase:
                                assigned_worker = w
                                min_cost_increase = cost_increase
                            w.remove_task(t.id)

                        else:
                            pass
                    if assigned_worker: 
                        assigned_worker.assign_task(t)
                        # Update self.assigned and self.unassigned
                        self.assigned.append(t.id)
                        self.unassigned.remove(t)
        
        self1 = self.copy()
        self2 = self.copy()
        find_best_assignment(self1, None, 0)
        find_best_assignment(self2, None, 1)
        
        if self1.objective() < self2.objective():
            self.workers = self1.workers
            self.assigned = self1.assigned
            self.unassigned = self1.unassigned
            self.skill_task_d = self1.skill_task_d
        else: 
            self.workers = self2.workers
            self.assigned = self2.assigned
            self.unassigned = self2.unassigned
            self.skill_task_d = self2.skill_task_d

        # Update self.solution 
        self.solution = [w.tasks_assigned for w in self.workers]
    
    def copy(self):
        return copy.deepcopy(self)
       
    
    def objective(self):
        ''' Calculate the objective value of the state
        Return the total cost of each worker + unassigned cost
        '''
        f1 = len(self.unassigned)
        f2 = sum(worker.get_objective() for worker in self.workers)
        return self.Alpha * f1 + f2


        
            




            

        