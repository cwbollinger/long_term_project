#!/usr/bin/env python

import sys
import os
import subprocess

import rospy
import actionlib
from long_term_server.msg import *
from long_term_server.srv import *
from std_srvs.srv import Empty


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
            return resp1.assigned_name 
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


class TaskActionServer(object):
    # create messages that are used to publish feedback/result
    _feedback = TaskFeedback()
    _result = TaskResult()

    def __init__(self, name):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, TaskAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
      
    def execute_cb(self, goal):
        # helper variables
        success = True
        print goal

        p = subprocess.Popen([os.path.expanduser('~/{}/devel/env.sh').format(goal.workspace_name), 'roslaunch', goal.package_name, goal.launchfile_name])

        self._feedback.status = "This is a test msg."
        self._as.publish_feedback(self._feedback)
        
        # start executing the action
        while  p.poll() is None:
            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                p.kill()
                success = False
                break
          
        if success:
            self._result.success_msg = "Hooray!"
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)
        
if __name__ == "__main__":
    server_client = LongTermAgentClient()
    agent_name = server_client.register_agent(sys.argv[1], sys.argv[1])
    namespace = '{}_agent'.format(agent_name)
    rospy.init_node('{}'.format(namespace))
    task_interface = TaskActionServer(namespace)
    rospy.spin()
