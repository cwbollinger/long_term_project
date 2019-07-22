#!/usr/bin/env python

import os
import signal
import subprocess
import importlib
import base64
import traceback

import rospy
from long_term_deployment.msg import AgentDescription, Task, TaskGoal, TaskFeedback, TaskResult, TaskAction
from long_term_deployment.srv import RegisterAgent, UnregisterAgent, GetRegisteredAgents
from long_term_deployment.synchronized_actions import SynchronizedActionClient, SynchronizedActionServer, SynchronizedSimpleActionServer

from actionlib_msgs.msg import GoalStatus

from std_msgs.msg import String
import threading
import Queue


class LongTermAgentClient(object):

    def __init__(self):
        rospy.loginfo('Waiting for services...')
        rospy.wait_for_service('/task_server/register_agent')
        self.register_agent_proxy = rospy.ServiceProxy(
            '/task_server/register_agent',
            RegisterAgent)
        rospy.wait_for_service('/task_server/unregister_agent')
        self.unregister_agent_proxy = rospy.ServiceProxy(
            '/task_server/unregister_agent',
            UnregisterAgent)
        rospy.wait_for_service('/task_server/get_agents')
        self.get_agents_proxy = rospy.ServiceProxy(
            '/task_server/get_agents',
            GetRegisteredAgents)
        rospy.loginfo('Services found!')

    def register_agent(self, a_name, a_type):
        description = AgentDescription()
        description.agent_name = a_name
        description.agent_type = a_type
        try:
            resp1 = self.register_agent_proxy(description)
            return resp1.assigned_name
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))
            return False

    def unregister_agent(self, a_name):
        try:
            resp1 = self.unregister_agent_proxy(a_name)
            return resp1.success
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))
            return False

    def get_agents(self):
        try:
            resp1 = self.get_agents_proxy()
            return resp1.agents
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))
            return []


class TaskActionServer(object):
    # create messages that are used to publish feedback/result
    _feedback = TaskFeedback()
    _result = TaskResult()

    def __init__(self):
        # load params for this robot/client into dict
        self.client_params = rospy.get_param('/client_params')
        rospy.loginfo(self.client_params)

        rospy.loginfo('Action Server Init')
        # Active/Primary Task server
        self.feedback_sub = rospy.Subscriber(
            '~active_feedback',
            String,
            self.update_active_feedback)
        self._action_name = name
        self._as = SynchronizedSimpleActionServer(
            "~active",
            TaskAction,
            execute_cb=self.execute_cb)

        # Continous Task tracking/server
        self.continuous_lock = threading.RLock()
        self._as_continuous = SynchronizedActionServer(
            "~continuous",
            TaskAction,
            self.start_continuous_task,
            self.stop_continuous_task)

        rospy.loginfo('connecting to own action interfaces...')
        self.continuous_client = SynchronizedActionClient(
            '~continuous',
            TaskAction)
        rospy.loginfo('connected!')

        # find the workspace so we can get tasks later
        current_path = os.path.abspath(__file__)
        # pkg_name = rospkg.get_package_name(current_path)
        ws_name = current_path.split('src/')[0]
        self.ws_name = ws_name[:-1]

    def start_task_thread(self, lf_handle, task_name, task_script, args, stop_event, result_queue):
        func = getattr(task_script, 'main')

        # make the main() in each task less nasty
        def t_main(s, q):
            try:
                retval = func(stop_event, args, self.client_params)
                rospy.loginfo("Task return value: {}".format(retval))
                q.put((True, retval))
            except BaseException as e:
                # If ANY exception fires, assume failure.
                # Task mains should catch non-critical exception internally
                rospy.logerr( 'Exception in task {}: {}'.format(task_name, e))
                rospy.logerr(traceback.format_exc())
                self.stop_task_launchfile(lf_handle)
                q.put((False, None))

        task_thread = threading.Thread(
            target=t_main,
            args=(stop_event, result_queue))
        task_thread.start()

        return task_thread

    def start_task_launchfile(self, task):
        # NOTE: The line below is equivalent, but way nastier
        # self._as_continuous.publish_feedback(gh.get_goal_status(), feedback)
        if task.workspace_name == '':
            workspace_name = self.ws_name
        else:
            workspace_name = '~/{}'.format(task.workspace_name)

        launch_args = ['{}:={}'.format(k, v)
                       for k, v in self.client_params.items()]
        #rospy.logdebug('Launchfile Args: {}'.format(launch_args))
        cmdlist = [
            os.path.expanduser('{}/devel/env.sh').format(workspace_name),
            'roslaunch',
            task.package_name,
            '{}.launch'.format(task.launchfile_name)] + launch_args

        devnull = None if task.debug else open(os.devnull, 'w')
        rospy.logwarn('Launchfile Command Args: {}'.format(cmdlist))
        p = subprocess.Popen(cmdlist, stdout=devnull, stderr=devnull)
        return p, devnull

    def stop_task_launchfile(self, launchfile_handle):
        proc, outstream = launchfile_handle
        if proc.poll() is None:  # launchfile hasn't closed yet
            rospy.logdebug('shutting down launch file')
            proc.send_signal(signal.SIGINT)  # some processes need this
            for i in range(10):  # give it 10 seconds to close cleanly
                if proc.poll() is None:
                    rospy.sleep(1)
                else:
                    break

        if proc.poll() is None:  # launchfile STILL hasn't closed yet
            rospy.logwarn('shutting down launch file, KILL required')
            proc.kill()  # for real this time

        if outstream:
            outstream.close()

    def update_active_feedback(self, msg):
        """ update feedback message and immediately send it."""
        self._feedback.status = msg.data
        self._as.publish_feedback(self._feedback)

    @staticmethod
    def taskname_from_gh(gh):
        task = gh.get_goal().task
        return (task.package_name, task.launchfile_name)

    def start_continuous_task(self, gh):

        with self.continuous_lock:
            task_thread = threading.Thread(
                target=self.continuous_task_entry,
                args=(gh,))
            task_thread.start()

    def stop_continuous_task(self, gh):
        goal_id = gh.get_goal_id()
        with self.continuous_lock:
            if goal_id.id in self._as_continuous.goals:
                gh.set_cancel_requested()
            else:
                warnmsg = "Task {} doesn't seem to be running?"
                rospy.logwarn(warnmsg.format(goal_id.id))

    def continuous_task_entry(self, gh):
        # if accept is not done before launching dependency tasks
        # they will never start because the currently processing goal
        # (this task) is still in limbo
        gh.set_accepted()

        # get the list of required tasks from the script file
        task_name = self.taskname_from_gh(gh)
        rospy.loginfo('Continuous Task {}'.format(task_name))

        running_tasks = ['{}/{}'.format(*self.taskname_from_gh(running_gh))
                         for running_gh in self._as_continuous.goals.values()]

        try:
            task_script = importlib.import_module('{}.{}'.format(*task_name))
            required_tasks = getattr(task_script, 'required_tasks')
            required_tasks.keys() # make sure it's a dict
        except ImportError:
            required_tasks = {}
        except AttributeError:
            rospy.loginfo('required_tasks undefined or not a dictionary')
            required_tasks = {}

        for taskname in required_tasks.keys():
            if taskname not in running_tasks:
                rospy.logwarn('{} not in running tasks'.format(taskname))
                rospy.logwarn('running tasks are: {}'.format(running_tasks))
                package_name, launchfile_name = taskname.split('/')
                dep_task = Task(
                    workspace_name='',
                    package_name=package_name,
                    launchfile_name=launchfile_name,
                    args=required_tasks[taskname],
                    debug=False)
                cgh = self.continuous_client.send_goal(TaskGoal(dep_task))

                while cgh.get_goal_status() != GoalStatus.ACTIVE:
                    rospy.loginfo('waiting for dependency {} to spin up...'.format(taskname))
                    rospy.loginfo('Task Status: {}'.format(cgh.get_goal_status_text()))
                    rospy.sleep(0.5)

        rospy.loginfo('Starting {} Background Thread...'.format(task_name))
        feedback = TaskFeedback(status='Continuous Task Ping')
        goal = gh.get_goal()
        gh.publish_feedback(feedback)

        t = goal.task
        rospy.loginfo('starting {} launchfile...'.format(task_name))
        launchfile_handle = self.start_task_launchfile(t)

        r = rospy.Rate(10)
        feedback.status = 'Starting Continuous Task...'
        stopEvent = threading.Event()
        queue = Queue.Queue()

        has_script = True
        try:
            task_name = (t.package_name, t.launchfile_name)
            task_script = importlib.import_module('{}.{}'.format(*task_name))
        # Continuous tasks may not have a script component
        except ImportError as e:
            rospy.logwarn('task script for {} not loaded because:'.format(t.launchfile_name))
            rospy.logwarn(e)
            has_script = False

        success = True

        if has_script:
            task_thread = self.start_task_thread(
                launchfile_handle,
                t.launchfile_name,
                task_script,
                t.args,
                stopEvent,
                queue)

            while task_thread.isAlive():
                gh.publish_feedback(feedback)
                # check that preempt has not been requested by the client
                with self.continuous_lock:
                    if gh.get_goal_status().status == GoalStatus.PREEMPTING:
                        log_msg = '{}: Continuous Task Preempted'
                        rospy.loginfo(log_msg.format(self._action_name))
                        stopEvent.set()  # end main, we're done
                        del self._as_continuous.goals[gh.get_goal_id().id]
                        success = False
                        break
                r.sleep()

        else:
            while not rospy.is_shutdown():
                with self.continuous_lock:
                    if gh.get_goal_status().status == GoalStatus.PREEMPTING:
                        log_msg = '{}: Continuous Task Shutdown Requested'
                        rospy.loginfo(log_msg.format(self._action_name))
                        break

                    # if launchfile has closed itself, end
                    if launchfile_handle[0].poll() is not None:
                        break

        self.stop_task_launchfile(launchfile_handle)

        # TODO: what should non-script launchfiles return on success?
        if has_script and success:
            success, result = queue.get()

        if has_script and success:
            # get main result, since it finished
            result = str(result)

            logmsg = '{}: Continuous Task {} Succeeded'
            rospy.loginfo(
                logmsg.format(
                    self._action_name,
                    goal.task.launchfile_name))

            rospy.loginfo('Result: {}'.format(result))
            rospy.loginfo('Queue len: {}'.format(queue.qsize()))
            gh.set_succeeded(
                result=base64.b64encode(result),
                text='task success!')

        elif success and not has_script:
            gh.set_succeeded(text='launchfile exited normally')

        elif not success and has_script:
            gh.set_canceled(text='preemption requested')

        else:
            gh.set_aborted(text='task failure')

    def execute_cb(self, goal):
        rospy.loginfo('New Task Received')
        r = rospy.Rate(10)
        stopEvent = threading.Event()
        queue = Queue.Queue()

        t = goal.task

        task_name = (t.package_name, t.launchfile_name)
        running_tasks = ['{}/{}'.format(*self.taskname_from_gh(gh))
                         for gh in self._as_continuous.goals.values()]
        # get the list of required tasks from the script file
        task_script = importlib.import_module('{}.{}'.format(*task_name))
        try:
            required_tasks = getattr(task_script, 'required_tasks')
        except ImportError:
            required_tasks = []
        except AttributeError:
            required_tasks = []

        rospy.loginfo('Required Dependencies: {}'.format(required_tasks))
        for x in required_tasks:
            if x not in running_tasks:
                rospy.loginfo('{} not in running tasks'.format(x))
                rospy.loginfo('running tasks are: {}'.format(running_tasks))
                package_name, launchfile_name = x.split('/')
                dep_task = Task(
                    workspace_name='',
                    package_name=package_name,
                    launchfile_name=launchfile_name,
                    args=[],
                    debug=False)
                self.continuous_client.send_goal(TaskGoal(dep_task))
                rospy.sleep(1)  # TODO: something better here...

        rospy.loginfo('starting active task launchfile...')
        launchfile_handle = self.start_task_launchfile(t)

        self._feedback.status = 'Starting...'
        success = True

        task_thread = self.start_task_thread(
            launchfile_handle,
            t.launchfile_name,
            task_script,
            t.args,
            stopEvent,
            queue)

        while task_thread.isAlive():
            self._as.publish_feedback(self._feedback)
            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                # trigger end of task main(), we're done
                stopEvent.set()
                success = False
                break
            r.sleep()

        # task is done, stop dependency nodes
        self.stop_task_launchfile(launchfile_handle)

        # if we didn't preempt, get the task's return state
        if success:
            success, result = queue.get()

        if success:
            result = str(result)
            # needed so json serialization works
            self._result.success_msg = base64.b64encode(result)

            rospy.loginfo('{}: Succeeded'.format(self._action_name))
            rospy.loginfo('Result: {}'.format(result))
            rospy.loginfo('Queue len: {}'.format(queue.qsize()))
            self._as.set_succeeded(self._result)


if __name__ == '__main__':
    rospy.init_node('robot_client', log_level=rospy.INFO)
    name = rospy.get_param('~agent_name', 'default')
    task_interface = TaskActionServer()
    try:
        server_client = LongTermAgentClient()
        agent_name = server_client.register_agent(name, name)

        def stop_agent():
            task_interface.continuous_lock.acquire()
            for gh in task_interface._as_continuous.goals.values():
                gh.set_cancel_requested()
            task_interface.continuous_lock.release()
            server_client.unregister_agent(agent_name)

        rospy.on_shutdown(stop_agent)
        rospy.spin()
    except rospy.exceptions.ROSInterruptException as e:
        rospy.logwarn(e)
        exit()
