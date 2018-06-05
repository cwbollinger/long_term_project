#!/usr/bin/env python

import collections

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus

from long_term_deployment.msg import *
from long_term_deployment.srv import *

Client = collections.namedtuple('Client', ['name', 'robot_type', 'action_client_name'])


class LongTermAgentServer(object):
    def __init__(self):
        self.agents = []
        self.task_queue = []
        rospy.init_node('task_server')
        name = rospy.get_name()
        self.s1 = rospy.Service('{}/register_agent'.format(name), RegisterAgent, self.handle_register_agent)
        self.s2 = rospy.Service('{}/unregister_agent'.format(name), UnregisterAgent, self.handle_unregister_agent)
        self.s3 = rospy.Service('{}/get_agents'.format(name), GetRegisteredAgents, self.handle_get_agents)
        self.s4 = rospy.Service('{}/queue_task'.format(name), QueueTask, self.queue_task)

    def main(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.schedule_tasks()
            rate.sleep()

    def handle_register_agent(self, req):
        if req.description not in self.agents:
            print('registering agent: {}'.format(req.description.agent_name))
            #action_client_name = actionlib.SimpleActionClient('{}_agent'.format(req.description.agent_name), TaskAction)
            action_client_name = '{}_agent'.format(req.description.agent_name)
            c = Client(req.description.agent_name, req.description.agent_type, action_client_name)
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
        return GetRegisteredAgentsResponse(self.agents)

    def queue_task(self, req):
        goal = TaskGoal()
        goal.workspace_name = req.workspace_name
        goal.package_name = req.package_name
        goal.launchfile_name = req.launchfile_name
        self.task_queue.append(goal)
        print('task queued...')
        return QueueTaskResponse(True)

    def schedule_tasks(self):
        if len(self.task_queue) == 0:
            #print('No tasks queued')
            return False

        for i, agent in enumerate(self.agents):
            client = actionlib.SimpleActionClient(agent.action_client_name, TaskAction)
            client.wait_for_server()
            status = client.simple_state
            if status == actionlib.SimpleGoalState.DONE or status == actionlib.SimpleGoalState.LOST:
                print('agent {} available'.format(agent.name))
                print('will wait until goal is complete...')
                goal = self.task_queue.pop(0)
                client.send_goal(goal)
                client.wait_for_result()
                print('Goal Complete!')
                return True

        print('All agents currently busy')
        return False


if __name__ == "__main__":
    task_server = LongTermAgentServer()
    print("Ready to register agents...")
    task_server.main()
