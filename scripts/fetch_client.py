#!/usr/bin/env python

import sys
import os
import subprocess

import rospy
import rospkg
import actionlib
from long_term_deployment.msg import *
from long_term_deployment.srv import *
from std_srvs.srv import Empty
from threading import Thread, RLock


class LongTermAgentClient(object):
    def __init__(self):
        print('Waiting for services...')
        rospy.wait_for_service('/task_server/register_agent')
        self.register_agent_proxy = rospy.ServiceProxy('/task_server/register_agent', RegisterAgent)
        rospy.wait_for_service('/task_server/unregister_agent')
        self.unregister_agent_proxy = rospy.ServiceProxy('/task_server/unregister_agent', UnregisterAgent)
        rospy.wait_for_service('/task_server/get_agents')
        self.get_agents_proxy = rospy.ServiceProxy('/task_server/get_agents', GetRegisteredAgents)
        print('Services found!')

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
        print('Action Server Init')
        self._action_name = name
        self._as = actionlib.SimpleActionServer("{}/active".format(self._action_name), TaskAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        # Continous Task tracking/server
        self.continuous_tasks = {}
        self.continuous_lock = RLock()
        self._as_continuous = actionlib.ActionServer("{}/continuous".format(self._action_name), TaskAction, self.start_continuous_task, self.stop_continuous_task, auto_start=False)
        self._as_continuous.start()

        current_path = os.path.abspath(__file__)
        pkg_name = rospkg.get_package_name(current_path)
        ws_name = current_path.split('src/{}'.format(pkg_name))[0]
        self.ws_name = os.path.split(ws_name[:-1])[1]

    def start_continuous_task(self, gh):
        self.continuous_lock.acquire()
        self.continuous_tasks[gh.get_goal_id()] = False
        self.continuous_lock.release()
        task_thread = Thread(target = self.continuous_task_entry, args = (gh,))
        task_thread.start()

    def stop_continuous_task(self, gh):
        self.continuous_lock.acquire()
        self.continuous_tasks[gh.get_goal_id()] = True
        self.continuous_lock.release()

    def continuous_task_entry(gh):
        success = True
        print('Incoming Continuous Task...')
        feedback = TaskFeedback(status="Continuous Task Ping")
        result = TaskFeedback(success_msg="It works?")
        gh.set_accepted()
        goal = gh.get_goal()
        self._as_continuous.publish_feedback(feedback)

        workspace_name = goal.workspace_name if goal.workspace_name != '' else self.ws_name

        # start executing the action
        p = subprocess.Popen([os.path.expanduser('~/{}/devel/env.sh').format(workspace_name), 'roslaunch', goal.package_name, goal.launchfile_name])

        r = rospy.Rate(10)
        while  p.poll() is None:
            gh.publish_feedback(feedback)
            # check that preempt has not been requested by the client
            self.continuous_lock.acquire()
            goal_id = gh.get_goal_id()
            if self.continuous_tasks[goal_id] == True:
                rospy.loginfo('{}: Preempted Task {}'.format(self._action_name, gh.get_goal().launchfile_name))
                result.success_msg="Got preempted"
                gh.set_canceled(result, "Preempt came in")
                p.kill()
                success = False
                del self.continuous_tasks[goal_id]
                self.continuous_lock.release()
                break
            r.sleep()

        if success:
            gh.set_succeeded(result)
            rospy.loginfo('{}: Succeeded'.format(gh.get_goal().launchfile_name))

    def execute_cb(self, goal):
        print('Incoming task...')
        success = True
        print goal

        self._feedback.status = "Ping"
        self._as.publish_feedback(self._feedback)
        workspace_name = goal.workspace_name if goal.workspace_name != '' else self.ws_name

        # start executing the action
        p = subprocess.Popen([os.path.expanduser('~/{}/devel/env.sh').format(workspace_name), 'roslaunch', goal.package_name, goal.launchfile_name])

        r = rospy.Rate(10)
        while  p.poll() is None:
            self._as.publish_feedback(self._feedback)
            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                p.kill()
                success = False
                break
            r.sleep()

        if success:
            self._result.success_msg = "Hooray!"
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)


if __name__ == "__main__":
    name = 'fetch'
    server_client = LongTermAgentClient()
    agent_name = server_client.register_agent(name, name)
    namespace = '{}_agent'.format(agent_name)
    rospy.init_node('{}'.format(namespace))
    task_interface = TaskActionServer(namespace)
    def stop_agent():
        task_interface.continuous_lock.acquire()
        for task in task_interface.continuous_tasks:
            task_interface.continuous_tasks[task] = True
        task_interface.continuous_lock.release()
        server_client.unregister_agent(agent_name)
    rospy.on_shutdown(stop_agent)
    rospy.spin()
