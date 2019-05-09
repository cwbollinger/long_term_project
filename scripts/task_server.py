#!/usr/bin/env python

import base64

from actionlib import SimpleGoalState

from actionlib_msgs.msg import GoalStatus

from long_term_deployment.msg import (
    AgentDescription,
    AgentStatus,
    Task,
    TaskAction,
    TaskGoal
)
from long_term_deployment.synchronized_actions import (
    SynchronizedActionClient,
    SynchronizedSimpleActionClient,
    get_task_from_gh
)

from long_term_deployment.srv import (
    AgentStatusList, AgentStatusListResponse,
    GetRegisteredAgents, GetRegisteredAgentsResponse,
    QueueTask, QueueTaskResponse,
    QueueTaskList, QueueTaskListResponse,
    RegisterAgent, RegisterAgentResponse,
    UnregisterAgent, UnregisterAgentResponse
)

from namedlist import namedlist

import rospy

Client = namedlist('Client', ['name', 'robot_type', 'active_action_client', 'background_action_client', 'last_ping_time'])


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
        self.s5 = rospy.Service('~start_continuous_task', QueueTask, self.start_continuous_task)
        self.s6 = rospy.Service('~stop_continuous_task', QueueTask, self.stop_continuous_task)
        self.s7 = rospy.Service('~get_queued_tasks', QueueTaskList, self.get_queued_tasks)
        self.s8 = rospy.Service('~get_agents_status', AgentStatusList, self.get_agents_status)

    def main(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.schedule_tasks()
            self.check_task_status()
            self.clear_disconnected_agents()
            rate.sleep()

    def handle_register_agent(self, req):
        if req.description not in self.agents:
            print('registering agent: {}'.format(req.description.agent_name))
            c = Client(req.description.agent_name, req.description.agent_type, None, None, rospy.get_time())
            self.agents.append(c)
            return RegisterAgentResponse(True, req.description.agent_name)
        return RegisterAgentResponse(False, '')

    def handle_unregister_agent(self, req):
        names = [a.name for a in self.agents]
        if req.agent_name in names:
            print('unregistering agent: "{}"'.format(req.agent_name))
            agent = self.agents[names.index(req.agent_name)]
            active_task = get_task_from_gh(agent.active_action_client.client.gh)
            if active_task is not None:
                self.task_queue.append(active_task)  # recover task so it is not lost
            del self.agents[names.index(req.agent_name)]
        return UnregisterAgentResponse(True)  # always succeed for now

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
            agent.last_ping_time = rospy.get_time()
            agent.background_action_client.send_goal(TaskGoal(req.task), feedback_cb=self.cb_creator(agent))
            return QueueTaskResponse(True)
        else:
            return QueueTaskResponse(False)

    def stop_continuous_task(self, req):
        agent = next((a for a in self.agents if a.name == req.agent.agent_name), None)

        if agent is None:
            return QueueTaskResponse(False)

        task_key = str(req.task)
        for gh in agent.background_action_client.goals:
            task = get_task_from_gh(gh)
            if task_key == str(task):
                # print(dir(gh))
                # gh.set_canceled()
                gh.cancel()
                return QueueTaskResponse(True)

        return QueueTaskResponse(False)

    def queue_task(self, req):
        # tasks can optionally specify a specific agent to assign to
        names = [a.name for a in self.agents]
        if req.agent.agent_name in names:
            agent = self.agents[names.index(req.agent.agent_name)]
            if agent.active_action_client is None:
                print('agent {} not initialized yet, cannot assign task...'.format(agent.name))
                return QueueTaskResponse(False)
            status = agent.active_action_client.get_state()
            if status not in self.TERMINAL_STATES:
                print('agent {} not available, currently busy'.format(agent.name))
                # print('will wait until goal is complete...')
            else:
                agent.last_ping_time = rospy.get_time()
                agent.active_action_client.send_goal(TaskGoal(req.task), feedback_cb=self.cb_creator(agent))
                print('Goal Sent!')
        else:
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
            if a.active_action_client is None:
                active_task = None
            else:
                active_task = get_task_from_gh(a.active_action_client.client.gh)
            status.active_task = active_task if active_task is not None else Task()
            status.background_tasks = []
            if a.background_action_client is not None:
                status.background_tasks = []
                for gh in a.background_action_client.goals:
                    task = get_task_from_gh(gh)
                    if task is None:
                        task = Task()
                    status.background_tasks.append(task)

            agents.append(status)
        return AgentStatusListResponse(agents)

    def schedule_tasks(self):
        for i, agent in enumerate(self.agents):
            if len(self.task_queue) == 0:
                return
            if agent.active_action_client is None:
                print('agent {} not initialized yet, skip for now...'.format(agent.name))
                continue  # not initialized fully, move on

            status = agent.active_action_client.get_state()
            if status in self.TERMINAL_STATES:
                print('agent {} available'.format(agent.name))
                # print('will wait until goal is complete...')
                active_task = self.task_queue.pop(0)
                agent.last_ping_time = rospy.get_time()
                agent.active_action_client.send_goal(TaskGoal(active_task), feedback_cb=self.cb_creator(agent))
                print('Goal Sent!')
                # agent.active_action_client.wait_for_result()
                # print('Result Complete!')

    def cb_creator(self, agent):
        def cb(*args):
            agent.last_ping_time = rospy.get_time()
        return cb

    def clear_disconnected_agents(self):
        t = rospy.get_time()

        for i, agent in enumerate(self.agents):
            if agent.active_action_client.client.simple_state == SimpleGoalState.DONE:
                continue

            active_task = get_task_from_gh(agent.active_action_client.client.gh)
            # print('{}: {:.3f}s since last ping'.format(agent.name, t-agent.last_ping_time))
            if t - agent.last_ping_time > 10:
                msg = '{} seems disconnected, requeueing task and removing agent from pool'
                print(msg.format(agent.name))
                self.task_queue.append(active_task)  # recover task so it is not lost
                del self.agents[i]

    def check_task_status(self):

        for agent in self.agents:
            if agent.active_action_client is None:  # this should run once per agent
                print('agent setup for {}'.format(agent.name))
                action_client_name = '{}_agent'.format(agent.name)
                print('waiting for active server')
                agent.active_action_client = SynchronizedSimpleActionClient(action_client_name + '/active', TaskAction)
                print('server found')
                print('waiting for continuous server')
                agent.background_action_client = SynchronizedActionClient(
                    action_client_name + '/continuous',
                    TaskAction)
                print('server found')

            status = agent.active_action_client.get_state()
            if status == GoalStatus.SUCCEEDED:
                result = agent.active_action_client.get_result()
                print('Result Returned:')
                print(base64.b64decode(result.success_msg))
                agent.active_action_client.stop_tracking_goal()

            if len(agent.background_action_client.goals) > 0:
                print(agent.name)

            for idx, gh in enumerate(agent.background_action_client.goals):
                task = get_task_from_gh(gh)
                if(task is None):
                    launchfile_name = 'Unknown? (gh={})'.format(gh)
                else:
                    launchfile_name = task.launchfile_name
                status = gh.get_goal_status()
                status_str = None
                if status in self.TERMINAL_STATES:
                    status_str = 'TERMINATED: {}'.format(gh.get_goal_status_text())
                    del agent.background_action_client.goals[idx]
                elif status == 0:
                    status_str = 'PENDING'
                elif status == 1:
                    status_str = 'ACTIVE'
                elif status == 6:
                    status_str = 'PREEMPTING'
                else:
                    status_str = 'Something else? {}'.format(status)

                print('\t{}: {}'.format(launchfile_name, status_str))


if __name__ == '__main__':
    task_server = LongTermAgentServer()
    print('Ready to register agents...')
    task_server.main()
