#!/usr/bin/env python

import rospy

from long_term_server.srv import *
from long_term_server.msg import *


class LongTermAgentServer(object):
    def __init__(self):
        self.agents = []
        self.nh = rospy.init_node('register_agent_server')
        self.s1 = rospy.Service('register_agent', RegisterAgent, self.handle_register_agent)
        self.s2 = rospy.Service('unregister_agent', UnregisterAgent, self.handle_unregister_agent)
        self.s3 = rospy.Service('get_agents', GetRegisteredAgents, self.handle_get_agents)

    def handle_register_agent(self, req):
        if req.description not in self.agents:
            self.agents.append(req.description)
        return RegisterAgentResponse(True) # always succeed for now

    def handle_unregister_agent(self, req):
        a_name = req.agent_name
        names = [a.agent_name for a in self.agents]
        if a_name in names:
            del self.agents[names.index(a_name)]
        return UnregisterAgentResponse(True) # always succeed for now

    def handle_get_agents(self, req):
        return GetRegisteredAgentsResponse(self.agents)
    

if __name__ == "__main__":
    server = LongTermAgentServer()
    print "Ready to register agents..."
    rospy.spin()
