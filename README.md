# ALNS_TimetableOptimization


## Problem Statement

Solve the Gig Worker Scheduling Problem (GSP) for a large F&B restaurant, defined as follows.   
 
Input: 
-	A planning horizon T days, each day is divided into hourly timeslots 0,…, 23 (representing 12 midnight to 11pm on the same day) 
-	A set of m hourly tasks to be performed, each task is associated with the timeslot (between 0 to 23), and the skill required
-	A set of n gig workers, each worker is associated with his/her available timeslots, a set of skills possessed, and an hourly rate

Output: A schedule which assigns tasks to workers in the entire horizon T subject to the following constraints:

1)	Each task is assigned to at most one worker.
2)	Each worker is assigned at most one task at the same time (i.e. no multi-tasking).
3)	Eligibility constraint: A task may only be assigned to a worker if the start time is in the worker’s available timeslots, and the skill required can be matched by the worker’s skill set. 
4)	Block constraint: A worker is turned out to perform tasks in time blocks. A block is defined as a period of consecutive time slots (e.g. Mon 9am-12noon). Each worker is allocated at most one block per day, and the maximum length of each block is BMax hours (e.g. BMax=12). 
5)	For each worker, the allocated blocks over the entire planning period must satisfy the following labor requirements:
a.	Maximum work constraint: The sum of the block lengths must not exceed WMax hours (e.g. WMax=72).
b.	Rest constraint: Between two blocks, there must be a rest period of at least RMin hours (e.g. RMin=12). In other words, the duration between the last assigned task of the current day and the first assigned task of the next day must be at least RMin hours (for simplicity, assume RMin≤24).
  
Objective function: The goal is to find a schedule that minimizs the cost function f1+ f2, where:

1)	f1: Total cost of unassigned tasks, calculated as 'alpha' times the number of unassigned tasks. (You can think of 'alpha'  as the monetary penalty cost for each unassigned task)
2)	f2: Total cost of workers, calculated as follows:
For each worker, the cost of each allocated block is the block length multiplied by the worker’s hourly rate, subject to a minimum value of 4 hours multiplied by the hourly rate. The rationale is that each time a worker turns out for work, he/she should be compensated with a base payment. The total cost of workers is the sum over the costs of allocated blocks. 
