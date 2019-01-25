#!/usr/bin/env python

import rospy
from long_term_deployment.msg import AgentDescription, Task
from long_term_deployment.srv import GetRegisteredAgents, QueueTask, QueueTaskList, AgentStatusList

# Position format is (x, y, theta)
tmp_dict = {'personal robotics lab': (1,1,0), 'graf 203': (3,3,1.57), "connors desk": (5,5,3.14)}

##############
## Functions to be called by Commands
## Contains desired robot behavior for each command (per chris' infrastructure of task scheduling)

# Look up target by name in Chris' map-coord dictionary
# TODO:  Currently assumes that 'target' is a valid key.  Allow target to /contain/ a valid key
def get_coords_from_name(target):
    return [str(x) for x in tmp_dict[target]]

# Set a nav goal at indicated target
def nav_to_target(target):
    global queue_active_task_proxy
    coords = get_coords_from_name(target)
    # Queue task 'go_to_pose' to the desired location, without debug, on the agent 'turtlebot'
    rospy.loginfo("Navigating to '{}' at {}".format(target, coords))
    queue_active_task_proxy(Task('','navigation_tasks','go_to_pose', coords, False), AgentDescription('turtlebot', 'turtlebot'))

# Rotate robot to point at indicated target
def get_target_coords(target):
    point_to_coords = get_coords_from_name(target)
    print("DEBUG MESSAGE:  Now turning to point at '" + target + "' at coordinates " + point_to_coords)

# Exhaustively searches the list potential_target to determine if a valid target
# substring (from target/coords dictionary) exists.  If so, return it, else -1
def search_for_valid_target(potential_target):
    substring_len = len(potential_target)
    for window_size in range(1,substring_len+1):
        for i in range(0,substring_len+1-window_size): 
            if ' '.join(potential_target[i:i+window_size]) in tmp_dict:
                return ' '.join(potential_target[i:i+window_size])
    return -1
            
#def queue_next_task(target):

##############
## Command families.  Single words that indicate a general type of work to do, 
## such as movement or information requests.
## When a Command is found as part of a task, run the corresponding function with arg 'Target'
task_type_dict = {}
task_func_dict = {}

# Words indicating a movement command, expected in tasks like 'go to Graf 203'
task_type_dict['c_go_to'] = ["go","bring","deliver"]
task_func_dict['c_go_to'] = nav_to_target

# Words indicative of an information request, as in 'where is Graf 205?'
task_type_dict['c_location_request'] = ["where", "location"]
task_func_dict['c_location_request'] = get_target_coords

# TODO, alias.  Make new dict entry 'whatever string' pointing to coords of a valid entry

# TODO, set location.  Make new dict entry 'whatever' pointing to current coords

# TODO, multi-task sequencing.  "go to Graf 203, then go to Rogers"
#task_type_dict['c_sequencing'] = ["and","then"]
#task_func_dict['c_sequencing'] = queue_next_task

# filler and linking words that will be ignored 
task_type_dict['s_ignore'] = ["the", "to", "is", "me", "this", "of"]


##############
## Main parser
## Parse input into task-relevant (or irrelevant) tokens
def parse_input_text(in_txt):
    request_tokens = in_txt.lower().split(" ")
    #print(request_tokens)
    for word_ind in range(0,len(request_tokens)):
        for keyword in task_type_dict:
            if request_tokens[word_ind] in task_type_dict[keyword]:
                request_tokens[word_ind] = keyword
                break

    #print(request_tokens)

    # Clean out ignored words (less hacky than handling below)
    request_tokens[:] = (i for i in request_tokens if i != 's_ignore') 

    #print(request_tokens)

    ## Task assembly.  Each valid task requires a Command and a Target
    task = []
    for word_ind in range(0,len(request_tokens)):
        if request_tokens[word_ind][:2] == 'c_':
            potential_target = request_tokens[word_ind+1:]
            target = search_for_valid_target(potential_target)
            if target is -1:
                print("ERROR:  No valid target found within input '" + ' '.join(potential_target) + "'")
            else:
                task_func_dict[request_tokens[word_ind]](target)

def main():
    robot_name = 'turtlebot'
    ##############
    ## Interactive main loop
    print('Use Ctrl-c to close this program\n')
    print("ROBOT '{}' is now accepting commands".format(robot_name))
    try:
        while True:
            in_str = raw_input('> ')
            if in_str == 'exit':
                break
            print('parsing...')
            parse_input_text(in_str)
    except KeyboardInterrupt:
        print("\nROBOT '{}' is no longer listening to you...".format(robot_name))

if __name__ == "__main__":
    rospy.init_node('server_terminal')
    rospy.wait_for_service('/task_server/get_agents')
    agents_proxy = rospy.ServiceProxy('/task_server/get_agents', GetRegisteredAgents)
    rospy.wait_for_service('/task_server/get_queued_tasks')
    queued_tasks_proxy = rospy.ServiceProxy('/task_server/get_queued_tasks', QueueTaskList)
    rospy.wait_for_service('/task_server/get_agents_status')
    active_tasks_proxy = rospy.ServiceProxy('/task_server/get_agents_status', AgentStatusList)
    rospy.wait_for_service('/task_server/queue_task')
    queue_active_task_proxy = rospy.ServiceProxy('/task_server/queue_task', QueueTask)

    main()

