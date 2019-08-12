#!/usr/bin/env python

import subprocess
from os.path import expanduser

import rospy
from long_term_deployment.msg import AgentDescription, Task
from long_term_deployment.srv import GetRegisteredAgents, QueueTask, QueueTaskList, AgentStatusList


class BackupManager(object):
    # TODO: Change this hard coded garbage to something reasonable
    login_data = {
        'erratic': 'tank@tank.engr.oregonstate.edu'
    }

    def __init__(self):
        rospy.wait_for_service('/task_server/get_agents')
        self.agents_proxy = rospy.ServiceProxy('/task_server/get_agents', GetRegisteredAgents)
        rospy.wait_for_service('/task_server/get_queued_tasks')
        self.queued_tasks_proxy = rospy.ServiceProxy('/task_server/get_queued_tasks', QueueTaskList)
        rospy.wait_for_service('/task_server/get_agents_status')
        self.active_tasks_proxy = rospy.ServiceProxy('/task_server/get_agents_status', AgentStatusList)
        rospy.wait_for_service('/task_server/queue_task')
        self.queue_active_task_proxy = rospy.ServiceProxy('/task_server/queue_task', QueueTask)
        rospy.wait_for_service('/task_server/start_continuous_task')
        self.start_continuous_task_proxy = rospy.ServiceProxy('/task_server/start_continuous_task', QueueTask)
        rospy.wait_for_service('/task_server/stop_continuous_task')
        self.stop_continuous_task_proxy = rospy.ServiceProxy('/task_server/stop_continuous_task', QueueTask)

    def main(self):
    
        r = rospy.Rate(0.01)
        while not rospy.is_shutdown():

            # Initialization
            self.agents = [a.agent_name for a in self.agents_proxy().agents]
            self.queued_tasks = ['{}/{}'.format(t.package_name, t.launchfile_name)
                            for t in self.queued_tasks_proxy().tasks]
            self.agent_tasks = {}
            for a in self.active_tasks_proxy().agent_statuses:
                self.agent_tasks[a.agent.agent_name] = {
                    'active_task': a.active_task.launchfile_name,
                    'background_tasks': [t.launchfile_name for t in a.background_tasks],
                }

            for agent in self.agent_tasks.keys():
                if self.agent_tasks[agent]['active_task'] == '':
                    if agent not in self.login_data:
                        rospy.logwarn("No login info provide for robot {}, cannot sync data".format(agent))
                    else:
                        bagpath = expanduser('~/bags')
                        cmd = ['rsync', '-r', '-v', '-e', 'ssh', '{}:~/bags/'.format(self.login_data[agent]), bagpath]
                        subprocess.check_call(cmd)
    
            r.sleep()


if __name__ == '__main__':
    rospy.init_node('server_terminal')
    node = BackupManager()
    node.main()

