import argparse
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import random
import copy

from gsp import Parser, GSP
from pathlib import Path

from alns import ALNS, State
from alns.criteria import HillClimbing


### save output file for the solution ###
def save_output(YourName, gsp, suffix):
    '''Generate output file (.txt) for the gsp solution, containing the instance name, the objective value, and the route
    Args:
        YourName::str
            your name, eg. John_Doe
        gsp::GSP
            a GSP object
        suffix::str
            suffix of the output file,
            eg. 'initial' for random initialization
            and 'solution' for the final solution
    '''
    workers = sorted(gsp.workers, key=lambda x:x.id)
    str_builder = [f'Objective: {gsp.objective()}, Unassigned: {[t.id for t in gsp.unassigned]}']
    str_builder += [str(w) for w in workers]
    str_builder = [e for e in str_builder if len(e)>0]
    with open('{}_{}_{}.txt'.format(YourName, gsp.name, suffix), 'w') as f:
        f.write('\n'.join(str_builder))

### Destroy operators ###

### Destroy operators ###

# Set degree of destruction
degree_of_destruction = 0.25

def num_task_to_remove(state):
    return int(len(state.tasks) * degree_of_destruction)

# You can follow the example and implement destroy_2, destroy_3, etc
def destroy_random_task(current:GSP, random_state):
    ''' Destroy operator sample (name of the function is free to change)
    Args:
        current::GSP
            a GSP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        destroyed::GSP
            the GSP object after destroying
    '''
    destroyed = current.copy()
    # You should code here
    destroy_list = random_state.choice(destroyed.assigned,num_task_to_remove(destroyed),replace=False)
    for task_id in destroy_list:
        for w in destroyed.workers:
            w.remove_task(task_id)
        # Update self.assigned
        destroyed.assigned.remove(task_id)
        # Update self.unassigned
        destroyed.unassigned.append(destroyed.taskID_taskObj[task_id])

    # Update self.solution
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]
    
    return destroyed 

def destroy_tasks_of_high_demand_skill(current:GSP, random_state):
    destroyed = current.copy()
    full_task_list = []
    assigned_tasks = []
    destroyed.skill_task_d = {k:v for k,v in sorted(destroyed.skill_task_d.items(), key=lambda x: len(x[1]), reverse=True)}
    for _,v in destroyed.skill_task_d.items():
        random.shuffle(v)
        full_task_list.extend(v)
    assigned_tasks = list(filter(lambda x: x.id in destroyed.assigned, full_task_list))
    destroy_list = assigned_tasks[:num_task_to_remove(destroyed)]

    for t in destroy_list:
        for w in destroyed.workers:
            w.remove_task(t.id)
        # Update self.assigned
        destroyed.assigned.remove(t.id)
        # Update self.unassigned
        destroyed.unassigned.append(t)

    # Update self.solution
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]
    
    return destroyed 

def destroy_tasks_by_hrs(current:GSP, random_state):
    destroyed = current.copy()
    hrs = set()
    for t in destroyed.unassigned:
        hrs.add(t.hour)
    # if there is no unassigned tasks, choose random hours to remove tasks from
    if not len(hrs):
        start_hr = random_state.choice(24-destroyed.workers[0].bmax+1,1,replace=False)[0]
        for h in range(start_hr,start_hr+destroyed.workers[0].bmax+1):
            hrs.add(h)
    destroy_list = list(filter(lambda x: x.id in destroyed.assigned and x.hour in hrs, destroyed.tasks))
    if len(destroy_list) > num_task_to_remove(destroyed):
        destroy_list = random_state.choice(destroy_list, num_task_to_remove(destroyed),replace=False)

    for t in destroy_list:
        for w in destroyed.workers:
            w.remove_task(t.id)
        # Update self.assigned
        destroyed.assigned.remove(t.id)
        # Update self.unassigned
        destroyed.unassigned.append(t)

    # Update self.solution
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]
    
    return destroyed 

def destroy_random_worker(current:GSP, random_state):
    destroyed = current.copy()
    assigned_worker = list(filter(lambda x: len(x.tasks_assigned)>0, destroyed.workers))
    destroy_worker = random_state.choice(assigned_worker,
                                        int(len(assigned_worker) * degree_of_destruction),replace=False)
    for w in destroy_worker:
        for task in w.tasks_assigned:
            w.remove_task(task.id)
            # Update self.assigned
            destroyed.assigned.remove(task.id)
            # Update self.unassigned
            destroyed.unassigned.append(destroyed.taskID_taskObj[task.id])

    # Update self.solution
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]
    
    return destroyed 

def destroy_most_task_worker(current:GSP, random_state):
    destroyed = current.copy()
    
    assigned_worker = list(filter(lambda x: len(x.tasks_assigned)>0, destroyed.workers))
    assigned_worker = sorted(assigned_worker,key=lambda x: len(x.tasks_assigned),reverse=True)
    destroy_worker = assigned_worker[:int(len(assigned_worker) * degree_of_destruction)]

    for w in destroy_worker:
        for task in w.tasks_assigned:
            w.remove_task(task.id)
            # Update self.assigned
            destroyed.assigned.remove(task.id)
            # Update self.unassigned
            destroyed.unassigned.append(destroyed.taskID_taskObj[task.id])

    # Update self.solution
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]
    
    return destroyed 
                                          
def destroy_least_task_worker(current:GSP, random_state):
    destroyed = current.copy()
    
    assigned_worker = list(filter(lambda x: len(x.tasks_assigned)>0, destroyed.workers))
    assigned_worker = sorted(assigned_worker,key=lambda x: len(x.tasks_assigned),reverse=False)
    destroy_worker = assigned_worker[:int(len(assigned_worker) * degree_of_destruction)]

    for w in destroy_worker:
        for task in w.tasks_assigned:
            w.remove_task(task.id)
            # Update self.assigned
            destroyed.assigned.remove(task.id)
            # Update self.unassigned
            destroyed.unassigned.append(destroyed.taskID_taskObj[task.id])

    # Update self.solution
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]
    
    return destroyed 

def destroy_high_cost_per_day_worker(current:GSP, random_state):
    destroyed = current.copy()
    
    assigned_worker = list(filter(lambda x: len(x.tasks_assigned)>0, destroyed.workers))
    assigned_worker = sorted(assigned_worker,key=lambda x: x.get_objective()/len(x.blocks.keys()),reverse=True)
    destroy_worker = assigned_worker[:int(len(assigned_worker) * degree_of_destruction)]

    for w in destroy_worker:
        for task in w.tasks_assigned:
            w.remove_task(task.id)
            # Update self.assigned
            destroyed.assigned.remove(task.id)
            # Update self.unassigned
            destroyed.unassigned.append(destroyed.taskID_taskObj[task.id])

    # Update self.solution
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]
    
    return destroyed 


def destroy_low_cost_per_day_worker(current:GSP, random_state):
    destroyed = current.copy()
    
    assigned_worker = list(filter(lambda x: len(x.tasks_assigned)>0, destroyed.workers))
    assigned_worker = sorted(assigned_worker,key=lambda x: x.get_objective()/len(x.blocks.keys()),reverse=False)
    destroy_worker = assigned_worker[:int(len(assigned_worker) * degree_of_destruction)]

    for w in destroy_worker:
        for task in w.tasks_assigned:
            w.remove_task(task.id)
            # Update self.assigned
            destroyed.assigned.remove(task.id)
            # Update self.unassigned
            destroyed.unassigned.append(destroyed.taskID_taskObj[task.id])

    # Update self.solution
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]
    
    return destroyed

def destroy_high_efficiency_worker(current:GSP, random_state):
    destroyed = current.copy()
    
    assigned_worker = list(filter(lambda x: len(x.tasks_assigned)>0, destroyed.workers))
    assigned_worker_least_efficient = list(filter(lambda x: len(x.tasks_assigned)/x.total_hours<=0.9, assigned_worker))

    task_time=[]
    destroy_list=[]
    for w in assigned_worker_least_efficient:
        for t in w.tasks_assigned:
            task_time.append((t.skill,t.hour,t.day))
            destroy_list.append(t)
    
    destroy_list2=[]
    for w in list(filter(lambda x: len(x.tasks_assigned)/x.total_hours>0.9, assigned_worker)):
        for t in w.tasks_assigned:
            if t not in destroy_list and (t.skill,t.hour,t.day) in task_time:
                destroy_list2.append(t)
            
    random.shuffle(destroy_list2)
    destroy_list.extend(destroy_list2)
    destroy_list = destroy_list[:num_task_to_remove(destroyed)]
            
    for t in destroy_list:
        for w in destroyed.workers:
            w.remove_task(t.id)
        # Update self.assigned
        destroyed.assigned.remove(t.id)
        # Update self.unassigned
        destroyed.unassigned.append(t)

    # Update self.solution
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]
    
    return destroyed 

def destroy_low_efficiency_worker(current:GSP, random_state):
    destroyed = current.copy()
    
    assigned_worker = list(filter(lambda x: len(x.tasks_assigned)>0, destroyed.workers))
    assigned_worker_most_efficient = list(filter(lambda x: len(x.tasks_assigned)/x.total_hours>0.9, assigned_worker))

    task_time=[]
    destroy_list=[]
    for w in assigned_worker_most_efficient:
        for t in w.tasks_assigned:
            task_time.append((t.skill,t.hour,t.day))
            destroy_list.append(t)
    
    destroy_list2=[]
    for w in list(filter(lambda x: len(x.tasks_assigned)/x.total_hours<=0.9, assigned_worker)):
        for t in w.tasks_assigned:
            if t not in destroy_list and (t.skill,t.hour,t.day) in task_time:
                destroy_list2.append(t)
            
    random.shuffle(destroy_list2)
    destroy_list.extend(destroy_list2)
    destroy_list = destroy_list[:num_task_to_remove(destroyed)]
            
    for t in destroy_list:
        for w in destroyed.workers:
            w.remove_task(t.id)
        # Update self.assigned
        destroyed.assigned.remove(t.id)
        # Update self.unassigned
        destroyed.unassigned.append(t)

    # Update self.solution
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]
    
    return destroyed 

    
### Repair operators ###
# You can follow the example and implement repair_2, repair_3, etc
def repair_random_func1(destroyed:GSP, random_state):
    ''' repair operator sample (name of the function is free to change)
    Args:
        destroyed::GSP
            a GSP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::GSP
            the GSP object after repairing
    '''
    # You should code here
    unassigned_tasks = destroyed.unassigned[:]
    random.shuffle(unassigned_tasks)
    for t in unassigned_tasks:
        # Assign t to a worker that incurs minimum increase in cost
        min_cost_increase = float('inf')
        assigned_worker = 0
        for w in destroyed.workers:
            if w.can_assign(t):
                # calculate cost increase
                cur_obj = w.get_objective()
                w.assign_task(t)
                updated_obj = w.get_objective()
                cost_increase = (updated_obj - cur_obj)/(w.rate*max(w.bmax-4,4))*(len(w.skills)/destroyed.number_skills) + (len(w.skills)/destroyed.number_skills)

                if cost_increase < min_cost_increase:
                    assigned_worker = w
                    min_cost_increase = cost_increase
                w.remove_task(t.id)
                
            else:
                pass
        if assigned_worker: 
            assigned_worker.assign_task(t)
            # Update self.assigned and self.unassigned
            destroyed.assigned.append(t.id)
            destroyed.unassigned.remove(t)
    # Update self.solution 
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]

    return destroyed

def repair_random_func2(destroyed:GSP, random_state):
    ''' repair operator sample (name of the function is free to change)
    Args:
        destroyed::GSP
            a GSP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::GSP
            the GSP object after repairing
    '''
    # You should code here
    unassigned_tasks = destroyed.unassigned[:]
    random.shuffle(unassigned_tasks)
    for t in unassigned_tasks:
        # Assign t to a worker that incurs minimum increase in cost
        min_cost_increase = float('inf')
        assigned_worker = 0
        for w in destroyed.workers:
            if w.can_assign(t):
                # calculate cost increase
                cur_obj = w.get_objective()
                w.assign_task(t)
                updated_obj = w.get_objective()
                
                utilization = len(w.tasks_assigned)/(w.blocks[t.day][1]-w.blocks[t.day][0]+1)    
                if t.day - 1 not in w.available and t.day + 1 not in w.available:
                    cost_increase = np.log((updated_obj - cur_obj +1)/(w.rate*max(w.bmax-4,4))*(len(w.skills)/destroyed.number_skills) *\
                                    1/utilization)
                elif t.day - 1 in w.available and t.day + 1 not in w.available:
                    cost_increase = np.log((updated_obj - cur_obj +1)/(w.rate*max(w.bmax-4,4))*(len(w.skills)/destroyed.number_skills) *\
                                    1/utilization *\
                                    max(w.available[t.day-1][1] - (23-(w.rmin-t.hour)),1)/24)
                elif t.day - 1 not in w.available and t.day + 1 in w.available:
                    cost_increase = np.log((updated_obj - cur_obj +1)/(w.rate*max(w.bmax-4,4))*(len(w.skills)/destroyed.number_skills) *\
                                    1/utilization *\
                                    max((w.rmin-(23-t.hour))- w.available[t.day+1][0] ,1)/24) 
                else:
                    cost_increase = np.log((updated_obj - cur_obj +1)/(w.rate*max(w.bmax-4,4))*(len(w.skills)/destroyed.number_skills) *\
                                    1/utilization *\
                                    (max(w.available[t.day-1][1] - (23-(w.rmin-t.hour)),1)/24 + \
                                    max((w.rmin-(23-t.hour))- w.available[t.day+1][0] ,1)/24))
                if cost_increase < min_cost_increase:
                    assigned_worker = w
                    min_cost_increase = cost_increase
                w.remove_task(t.id)
                
            else:
                pass
        if assigned_worker: 
            assigned_worker.assign_task(t)
            # Update self.assigned and self.unassigned
            destroyed.assigned.append(t.id)
            destroyed.unassigned.remove(t)
    # Update self.solution 
    destroyed.solution = [w.tasks_assigned for w in destroyed.workers]

    return destroyed
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data')
    parser.add_argument(dest='data', type=str, help='data')
    parser.add_argument(dest='seed', type=int, help='seed')
    args = parser.parse_args()
    
    # instance file and random seed
    json_file = args.data
    seed = int(args.seed)
    
    # load data and random seed
    parsed = Parser(json_file)
    gsp = GSP(parsed.name, parsed.workers, parsed.tasks, parsed.Alpha)
    
    # construct random initialized solution
    gsp.random_initialize(seed)
    
    print("Initial solution objective is {}.".format(gsp.objective()))
    
    # visualize initial solution and generate output file
    save_output('LeDuyHoang', gsp, 'initial')
    
    # ALNS
    random_state = rnd.RandomState(seed)

    alns = ALNS(random_state)
    # add destroy
    # You should add all your destroy and repair operators
    alns.add_destroy_operator(destroy_random_task)
    alns.add_destroy_operator(destroy_tasks_of_high_demand_skill)
    alns.add_destroy_operator(destroy_tasks_by_hrs)
    alns.add_destroy_operator(destroy_random_worker)
    alns.add_destroy_operator(destroy_most_task_worker)
    alns.add_destroy_operator(destroy_least_task_worker)
    alns.add_destroy_operator(destroy_high_cost_per_day_worker)
    alns.add_destroy_operator(destroy_low_cost_per_day_worker)
    alns.add_destroy_operator(destroy_high_efficiency_worker)
    alns.add_destroy_operator(destroy_low_efficiency_worker)

    # add repair
    alns.add_repair_operator(repair_random_func1)
    alns.add_repair_operator(repair_random_func2)

    
    # run ALNS
    # select cirterion
    criterion = HillClimbing()
    # assigning weights to methods
    omegas = [3, 2, 1, 0.5]
    lambda_ = 0.8
    result = alns.iterate(gsp, omegas, lambda_, criterion,
                          iterations=10000, collect_stats=True)

    # result
    solution = result.best_state
    objective = solution.objective()
    print('Best heuristic objective is {}.'.format(objective))
    
    # visualize final solution and generate output file
    save_output('LeDuyHoang', solution, 'solution')
    