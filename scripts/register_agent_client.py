#!/usr/bin/env python

import sys
import rospy
from long_term_server.msg import *
from long_term_server.srv import *


class LongTermAgentClient(object):
    def __init__(self):
        rospy.wait_for_service('register_agent')
        self.register_agent_proxy = rospy.ServiceProxy('register_agent', RegisterAgent)
        rospy.wait_for_service('unregister_agent')
        self.unregister_agent_proxy = rospy.ServiceProxy('unregister_agent', UnregisterAgent)
        rospy.wait_for_service('get_agents')
        self.get_agents_proxy = rospy.ServiceProxy('get_agents', GetRegisteredAgents)

    def register_agent(self, a_name, a_type):
        description = AgentDescription()
        description.agent_name = a_name
        description.agent_type = a_type
        try:
            resp1 = self.register_agent_proxy(description)
            if resp1.success:
                self.task_service = rospy.Service('start_task_{}'.format(a_name), StartTask, self.start_task)
            return resp1.success 
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
            return False 

    def unregister_agent(self, a_name):
        try:
            resp1 = self.unregister_agent_proxy(a_name)
            return resp1.success 
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
            return False 

    def get_agents(self):
        try:
            resp1 = self.get_agents_proxy()
            return resp1.agents
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
            return [] 


if __name__ == "__main__":
    client = LongTermAgentClient()
    client.register_agent('a1', 'turtle')
    client.register_agent('a2', 'turtle')
    client.register_agent('b1', 'potato')
    print client.get_agents()
    client.unregister_agent('a2')
    print client.get_agents()

