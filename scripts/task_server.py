#!/usr/bin/env python

import collections

import rospy
import actionlib

from long_term_server.msg import *
from long_term_server.srv import *

Client = collections.namedtuple('Client', ['name', 'robot_type', 'action_client'], )


class LongTermAgentServer(object):
    def __init__(self):
        self.agents = []
        rospy.init_node('task_server')
        name = rospy.get_name()
        self.s1 = rospy.Service('{}/register_agent'.format(name), RegisterAgent, self.handle_register_agent)
        self.s2 = rospy.Service('{}/unregister_agent'.format(name), UnregisterAgent, self.handle_unregister_agent)
        self.s3 = rospy.Service('{}/get_agents'.format(name), GetRegisteredAgents, self.handle_get_agents)
        self.s4 = rospy.Service('{}/run_task'.format(name), RunTask, self.send_task)

    def handle_register_agent(self, req):
        if req.description not in self.agents:
            print 'registering agent: {}'.format(req.description.agent_name)
            action_client = actionlib.SimpleActionClient('{}_agent'.format(req.description.agent_name), TaskAction)
            c = Client(req.description.agent_name, req.description.agent_type, action_client)
            self.agents.append(c)
            return RegisterAgentResponse(True, req.description.agent_name)
        return RegisterAgentResponse(False, "")

    def handle_unregister_agent(self, req):
        names = [a.name for a in self.agents]
        if a_name in names:
            print 'unregistering agent: "{}"'.format(a_name)
            del self.agents[names.index(a_name)]
        return UnregisterAgentResponse(True) # always succeed for now

    def handle_get_agents(self, req):
        return GetRegisteredAgentsResponse(self.agents)

    def send_task(self, req):
        names = [a.name for a in self.agents]
        goal = TaskGoal()
        goal.workspace_name = req.workspace_name
        goal.package_name = req.package_name
        goal.launchfile_name = req.launchfile_name
        idx = names.index(req.agent_name)
        agent = self.agents[idx]
        agent.action_client.send_goal(goal)
        agent.action_client.wait_for_result()
        print(agent.action_client.get_result())
        return True


if __name__ == "__main__":
    task_server = LongTermAgentServer()
    print "Ready to register agents..."
    rospy.spin()
