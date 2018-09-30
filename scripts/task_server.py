#!/usr/bin/env python

from namedlist import namedlist

import base64

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus

from std_msgs.msg import String
from long_term_deployment.msg import *
from long_term_deployment.srv import RegisterAgent, RegisterAgentResponse, UnregisterAgent, UnregisterAgentResponse, GetRegisteredAgents, GetRegisteredAgentsResponse, QueueTask, QueueTaskResponse, QueueTaskList, QueueTaskListResponse, AgentStatusList, AgentStatusListResponse

Client = namedlist('Client', ['name', 'robot_type', 'active_action_client', 'active_task', 'background_action_client', 'background_tasks', 'last_ping_time'])


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
        self.s1 = rospy.Service('~register_agent', RegisterAgent, self.handle_register_agent)
        self.s2 = rospy.Service('~unregister_agent', UnregisterAgent, self.handle_unregister_agent)
        self.s3 = rospy.Service('~get_agents', GetRegisteredAgents, self.handle_get_agents)
        self.s4 = rospy.Service('~queue_task', QueueTask, self.queue_task)
        self.s4 = rospy.Service('~start_continuous_task', QueueTask, self.start_continuous_task)
        self.s5 = rospy.Service('~get_queued_tasks', QueueTaskList, self.get_queued_tasks)
        self.s6 = rospy.Service('~get_agents_status', AgentStatusList, self.get_agents_status)

    def main(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.schedule_tasks()
            self.check_task_status()
            self.clear_dcd_agents()
            rate.sleep()

    def handle_register_agent(self, req):
        if req.description not in self.agents:
            print('registering agent: {}'.format(req.description.agent_name))
            c = Client(req.description.agent_name, req.description.agent_type, None, None, None, [], rospy.get_time())
            self.agents.append(c)
            return RegisterAgentResponse(True, req.description.agent_name)
        return RegisterAgentResponse(False, "")

    def handle_unregister_agent(self, req):
        names = [a.name for a in self.agents]
        if req.agent_name in names:
            print('unregistering agent: "{}"'.format(req.agent_name))
            agent = self.agents[names.index(req.agent_name)]
            if agent.active_task != None:
                self.task_queue.append(agent.active_task) # recover task so it is not lost
            del self.agents[names.index(req.agent_name)]
        return UnregisterAgentResponse(True) # always succeed for now

    def handle_get_agents(self, req):
        agents = []
        for a in self.agents:
            tmp = AgentDescription()
            tmp.agent_name = a.name
            tmp.agent_type = a.robot_type
            agents.append(tmp)
        return GetRegisteredAgentsResponse(agents)

    def start_continuous_task(self, req):
        agent = next((a for a in self.agents if a.name == req.agent.agent_name), None)
        if agent is not None:
            agent.background_tasks.append(req.task)
            agent.last_ping_time = rospy.get_time()
            agent.background_action_client.send_goal(TaskGoal(req.task), feedback_cb=self.cb_creator(agent))
            return QueueTaskResponse(True)
        else: 
            return QueueTaskResponse(False)

    def queue_task(self, req):
        self.task_queue.append(req.task)
        print('task queued...')
        return QueueTaskResponse(True)

    def get_queued_tasks(self, req):
        return QueueTaskListResponse(self.task_queue)

    def get_agents_status(self, req):
        agents = []
        for a in self.agents:
            status = AgentStatus()
            status.agent = AgentDescription(a.name, a.robot_type)
            status.active_task = a.active_task if a.active_task != None else Task()
            status.background_tasks = a.background_tasks
            agents.append(status)
        return AgentStatusListResponse(agents)

    def schedule_tasks(self):
        for i, agent in enumerate(self.agents):
            if len(self.task_queue) == 0:
                return
            if agent.active_action_client == None:
                print('agent {} not initialized yet, skip for now...'.format(agent.name))
                continue # not initialized fully, move on
            status = agent.active_action_client.get_state()
            if status in self.TERMINAL_STATES:
                print('agent {} available'.format(agent.name))
                #print('will wait until goal is complete...')
                agent.active_task = self.task_queue.pop(0)
                agent.last_ping_time = rospy.get_time()
                agent.active_action_client.send_goal(TaskGoal(agent.active_task), feedback_cb=self.cb_creator(agent))
                print('Goal Sent!')
                #agent.active_action_client.wait_for_result()
                #print('Result Complete!')

    def cb_creator(self, agent):
        def cb(*args):
            agent.last_ping_time = rospy.get_time()
        return cb

    def clear_dcd_agents(self):
        t = rospy.get_time()

        for i, agent in enumerate(self.agents):
            if agent.active_task != None:
                #print('{}: {:.3f}s since last ping'.format(agent.name, t-agent.last_ping_time))
                if t - agent.last_ping_time > 10:
                    print('{} seems disconnected, requeueing task and removing agent from pool'.format(agent.name))
                    self.task_queue.append(agent.active_task) # recover task so it is not lost
                    del self.agents[i]

    def check_task_status(self):

        for agent in self.agents:
            if agent.active_action_client == None: # this should run once per agent
                print('agent setup for {}'.format(agent.name))
                action_client_name = '{}_agent'.format(agent.name)
                agent.active_action_client = actionlib.SimpleActionClient(action_client_name+'/active', TaskAction)
                print('waiting for active server')
                agent.active_action_client.wait_for_server()
                print('server found')
                agent.background_action_client = actionlib.ActionClient(action_client_name+'/continuous', TaskAction)
                print('waiting for continuous server')
                agent.background_action_client.wait_for_server()
                print('server found')

            if agent.active_task != None:
                status = agent.active_action_client.get_state()
                if status in self.TERMINAL_STATES:
                    agent.active_task = None

                if status == GoalStatus.SUCCEEDED:
                    result = agent.active_action_client.get_result()
                    print('Result Returned:')
                    print(base64.b64decode(result.success_msg))


if __name__ == "__main__":
    task_server = LongTermAgentServer()
    print("Ready to register agents...")
    task_server.main()
