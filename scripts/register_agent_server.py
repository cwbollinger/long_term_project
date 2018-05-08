#!/usr/bin/env python

import collections

import rospy

from long_term_server.srv import *
from long_term_server.msg import *

Client = collections.namedtuple('Client', ['name', 'robot_type', 'action_client'], )


class LongTermAgentServer(object):
    def __init__(self):
        self.agents = []
        rospy.init_node('task_server')
        self.s1 = rospy.Service('register_agent', RegisterAgent, self.handle_register_agent)
        self.s2 = rospy.Service('unregister_agent', UnregisterAgent, self.handle_unregister_agent)
        self.s3 = rospy.Service('get_agents', GetRegisteredAgents, self.handle_get_agents)

    def handle_register_agent(self, req):
        if req.description not in self.agents:
            print 'registering agent: {}'.format(req.description.agent_name)
            c = Client(req.description.agent_name, req.description.agent_type, None)
            c.action_client = actionlib.SimpleActionClient
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

    def send_task(self, agent_name, workspace_name, package_name, launchfile_name):
        names = [a.name for a in self.agents]
        idx = names.index(agent_name)
        goal = TaskGoal()
        print(goal)
        return
        '''
        self.agents[idx].action_client.send_goal

        pass
        '''


if __name__ == "__main__":
    server = LongTermAgentServer()
    print "Ready to register agents..."
    rospy.spin()
