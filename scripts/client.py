#!/usr/bin/env python

import sys
import os
import signal
import subprocess
import importlib
import base64

import rospy
import rospkg
import actionlib
from actionlib_msgs.msg import GoalStatus
from long_term_deployment.msg import AgentDescription, Task, TaskFeedback, TaskResult, TaskAction
from long_term_deployment.srv import RegisterAgent, UnregisterAgent, GetRegisteredAgents
from std_msgs.msg import String
from std_srvs.srv import Empty
import threading
import Queue

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

    def __init__(self):
        self.client_params = rospy.get_param('/client_params')
        print(self.client_params)
        print('Action Server Init')
        self.feedback_sub = rospy.Subscriber('~active_feedback', String, self.update_active_feedback)
        self._action_name = name
        self._as = actionlib.SimpleActionServer("~active", TaskAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        # Continous Task tracking/server
        self.continuous_tasks = {}
        self.continuous_lock = threading.RLock()
        self._as_continuous = actionlib.ActionServer("~continuous", TaskAction, self.start_continuous_task, self.stop_continuous_task, auto_start=False)
        self._as_continuous.start()

        current_path = os.path.abspath(__file__)
        pkg_name = rospkg.get_package_name(current_path)
        ws_name = current_path.split('src/')[0]
        self.ws_name = ws_name[:-1]
        #print(self.ws_name)

    def update_active_feedback(self, msg):
        ''' update feedback message and immediately send it. '''
        self._feedback.status = msg.data
        self._as.publish_feedback(self._feedback)

    def start_continuous_task(self, gh):
        with self.continuous_lock:
            self.continuous_tasks[gh.get_goal_id()] = False
            task_thread = threading.Thread(target = self.continuous_task_entry, args = (gh,))
            task_thread.start()

    def stop_continuous_task(self, gh):
        goal_id = gh.get_goal_id()
        with self.continuous_lock:
            if goal_id in self.continuous_tasks:
                self.continuous_tasks[goal_id] = True
            else:
                warnmsg = 'Task {} doesn\'t seem to be running?'
                rospy.logwarn(warnmsg.format(goal_id))

    def continuous_task_entry(self, gh):
        success = True
        print('Incoming Continuous Task...')
        feedback = TaskFeedback(status="Continuous Task Ping")
        gh.set_accepted()
        goal = gh.get_goal()
        t = goal.task
        gh.publish_feedback(feedback)
        #self._as_continuous.publish_feedback(gh.get_goal_status(), feedback)
        if t.workspace_name == '':
            workspace_name = self.ws_name
        else:
            workspace_name = '~/{}'.format(t.workspace_name)

        #print('{}/devel/env.sh'.format(workspace_name))
        launch_args = ['{}:={}'.format(k, v) for k, v in self.client_params.items()]
        cmdlist = [os.path.expanduser('{}/devel/env.sh').format(workspace_name), 'roslaunch', t.package_name, "{}.launch".format(t.launchfile_name)] + launch_args
        print(cmdlist)

        if t.debug:
            devnull = None
        else:
            devnull = open(os.devnull, 'w')

        p = subprocess.Popen(cmdlist, stdout=devnull, stderr=devnull)

        r = rospy.Rate(10)
        feedback.status = "Starting Continuous Task..."
        stopEvent = threading.Event()
        queue = Queue.Queue()

        success = True
        has_script = True # we assume there's an active part of the "task"

        try:
            #print("{}.{}".format(t.package_name, t.launchfile_name))
            task = importlib.import_module("{}.{}".format(t.package_name, t.launchfile_name))
            #print(dir(task))
            func = getattr(task, 'main')

            def t_main(s, q): # make the main() in each task less nasty
                q.put(func(stopEvent, goal.task.args))
                return q

            task_thread = threading.Thread(target = t_main, args = (stopEvent, queue))
            task_thread.start()

            while task_thread.isAlive():
                gh.publish_feedback(feedback)
                # check that preempt has not been requested by the client
                with self.continuous_lock:
                    if self.continuous_tasks[gh.get_goal_id()]:
                        rospy.loginfo('%s: Continuous Task Preempted' % self._action_name)
                        stopEvent.set() # end main, we're done
                        success = False
                        break
                r.sleep()
        except ImportError as e: # Continuous tasks may not have a script component
            rospy.logdebug('task script not loaded because:')
            rospy.logdebug(e)
            has_script = False
            while not rospy.is_shutdown():
                with self.continuous_lock:
                    if self.continuous_tasks[gh.get_goal_id()]:
                        rospy.loginfo('{}: Continuous Task Shutdown Requested'.format(self._action_name))
                        stopEvent.set() # end main, we're done
                        break
                    if p.poll() is not None: # launchfile has exited
                        break

        if p.poll() is None: # launchfile hasn't closed yet
            rospy.logdebug('shutting down launch file')
            p.send_signal(signal.SIGINT) # some processes need this
            for i in range(10): # give it 10 seconds to close cleanly
                if p.poll() is None:
                    rospy.sleep(1)
                else:
                    break

        if p.poll() is None: # launchfile STILL hasn't closed yet
            rospy.logwarn('shutting down launch file, KILL required')
            p.kill() # for real this time

        if devnull:
            devnull.close() # Don't need this pipe redirect since process is kill

        if success and has_script: # TODO: what should non-script launchfiles return on success?
            result = str(queue.get())
            result = TaskFeedback(success_msg=result) # get main result, since it finished

            logmsg = '{}: Continuous Task {} Succeeded'
            rospy.loginfo(logmsg.format(self._action_name, goal.task.launchfile_name))
            rospy.loginfo('Result: {}'.format(result))
            rospy.loginfo('Queue len: {}'.format(queue.qsize()))
            gh.set_succeeded(result=base64.b64encode(result), text='task success!')

        elif success and not has_script:
            gh.set_succeeded(text='launchfile exited normally')

        elif not success and has_script:
            gh.set_canceled(text='preemption requested')

        else:
            gh.set_aborted(text='task failure')

    def execute_cb(self, goal):
        t = goal.task
        if t.workspace_name != '':
            workspace_name = '~/{}'.format(t.workspace_name)
        else:
            workspace_name = self.ws_name

        # Startup dependencies
        print(goal.task.package_name, goal.task.launchfile_name)
        launch_args = ['{}:={}'.format(k, v) for k, v in self.client_params.items()]
        cmdlist = [os.path.expanduser('{}/devel/env.sh').format(workspace_name), 'roslaunch', t.package_name, "{}.launch".format(t.launchfile_name)] + launch_args
        print(cmdlist)

        r = rospy.Rate(10)
        stopEvent = threading.Event()
        queue = Queue.Queue()

        if t.debug:
            devnull = None
        else:
            devnull = open(os.devnull, 'w')

        p = subprocess.Popen(cmdlist, stdout=devnull, stderr=devnull)

        task_script = importlib.import_module("{}.{}".format(t.package_name, t.launchfile_name))

        func = getattr(task_script, 'main')

        def t_main(s, q): # make the main() in each task less nasty
            q.put(func(stopEvent, t.args))
            #return q

        self._feedback.status = "Starting..."
        success = True

        task_thread = threading.Thread(target = t_main, args = (stopEvent, queue))
        task_thread.start()

        while task_thread.isAlive():
            self._as.publish_feedback(self._feedback)
            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                stopEvent.set() # trigger end of task main(), we're done
                success = False
                break
            r.sleep()

        if p.poll() is None: # launchfile hasn't closed yet
            p.kill() # close it

        if devnull:
            devnull.close()

        if success:
            result = str(queue.get()) # get result from main, since it finished
            self._result.success_msg = base64.b64encode(result) # needed so json serialization works            
            rospy.loginfo('{}: Succeeded'.format(self._action_name))
            rospy.loginfo('Result: {}'.format(result))
            rospy.loginfo('Queue len: {}'.format(queue.qsize()))
            self._as.set_succeeded(self._result)


if __name__ == "__main__":
    rospy.init_node('robot_client')
    name = rospy.get_param("~agent_name", "default")
    task_interface = TaskActionServer()
    server_client = LongTermAgentClient()
    agent_name = server_client.register_agent(name, name)

    def stop_agent():
        task_interface.continuous_lock.acquire()
        for task in task_interface.continuous_tasks:
            task_interface.continuous_tasks[task] = True
        task_interface.continuous_lock.release()
        server_client.unregister_agent(agent_name)

    rospy.on_shutdown(stop_agent)
    rospy.spin()
