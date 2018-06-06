#!/usr/bin/env python

from namedlist import namedlist

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus

from std_msgs.msg import String
from long_term_deployment.msg import *
from long_term_deployment.srv import *

Client = namedlist('Client', ['name', 'robot_type', 'action_client', 'active_task'])


class LongTermAgentServer(object):
    TERMINAL_STATES = [
        GoalStatus.LOST,
        GoalStatus.REJECTED,
        GoalStatus.RECALLED,
        GoalStatus.PREEMPTED,
        GoalStatus.ABORTED,
        GoalStatus.SUCCEEDED
    ]

    def __init__(self):
        self.agents = []
        self.task_queue = []
        rospy.init_node('task_server')
        name = rospy.get_name()
        self.s1 = rospy.Service('{}/register_agent'.format(name), RegisterAgent, self.handle_register_agent)
        self.s2 = rospy.Service('{}/unregister_agent'.format(name), UnregisterAgent, self.handle_unregister_agent)
        self.s3 = rospy.Service('{}/get_agents'.format(name), GetRegisteredAgents, self.handle_get_agents)
        self.s4 = rospy.Service('{}/queue_task'.format(name), QueueTask, self.queue_task)
        self.s5 = rospy.Service('{}/get_queued_tasks'.format(name), QueueTaskList, self.get_queued_tasks)
        self.s6 = rospy.Service('{}/get_active_tasks'.format(name), ActiveTaskList, self.get_active_tasks)

    def main(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.schedule_tasks()
            self.check_task_status()
            rate.sleep()

    def handle_register_agent(self, req):
        if req.description not in self.agents:
            print('registering agent: {}'.format(req.description.agent_name))
            c = Client(req.description.agent_name, req.description.agent_type, None, None)
            self.agents.append(c)
            return RegisterAgentResponse(True, req.description.agent_name)
        return RegisterAgentResponse(False, "")

    def handle_unregister_agent(self, req):
        names = [a.name for a in self.agents]
        if a_name in names:
            print('unregistering agent: "{}"'.format(a_name))
            del self.agents[names.index(a_name)]
        return UnregisterAgentResponse(True) # always succeed for now

    def handle_get_agents(self, req):
        agents = []
        for a in self.agents:
            tmp = AgentDescription()
            tmp.agent_name = a.name
            tmp.agent_type = a.robot_type
            agents.append(tmp)
        return GetRegisteredAgentsResponse(agents)

    def queue_task(self, req):
        goal = TaskGoal()
        goal.workspace_name = req.workspace_name
        goal.package_name = req.package_name
        goal.launchfile_name = req.launchfile_name
        self.task_queue.append(goal)
        print('task queued...')
        return QueueTaskResponse(True)

    def get_queued_tasks(self, req):
        readable_tasks = []
        for t in self.task_queue:
            readable_tasks.append(String(t.workspace_name+'/'+t.package_name+'/'+t.launchfile_name))
        return QueueTaskListResponse(readable_tasks)

    def get_active_tasks(self, req):
        active_agents = []
        active_tasks = []
        for a in self.agents:
            if a.active_task != None:
                active_agents.append(String(a.name))
                active_tasks.append(String(a.active_task))
        return ActiveTaskListResponse(active_agents, active_tasks)

    def schedule_tasks(self):
        if len(self.task_queue) == 0:
            #print('No tasks queued')
            return False

        for i, agent in enumerate(self.agents):
            if agent.action_client == None:
               continue # not initialized fully, move on
            status = agent.action_client.get_state()
            if status in self.TERMINAL_STATES:
                print('agent {} available'.format(agent.name))
                #print('will wait until goal is complete...')
                goal = self.task_queue.pop(0)
                agent.active_task = goal.workspace_name+'/'+goal.package_name+'/'+goal.launchfile_name
                agent.action_client.send_goal(goal)
                print('Goal Sent!')
                #agent.action_client.wait_for_result()
                #print('Result Complete!')

    def check_task_status(self):

        for agent in self.agents:
            if agent.action_client == None: # this should run once per agent
                action_client_name = '{}_agent'.format(agent.name)
                agent.action_client = actionlib.SimpleActionClient(action_client_name, TaskAction)
                agent.action_client.wait_for_server()

            if agent.active_task != None:
                status = agent.action_client.get_state()
                if status in self.TERMINAL_STATES:
                    agent.active_task = None


if __name__ == "__main__":
    task_server = LongTermAgentServer()
    print("Ready to register agents...")
    task_server.main()
