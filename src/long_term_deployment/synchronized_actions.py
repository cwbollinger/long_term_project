import weakref

import actionlib
from actionlib import ActionClient, ActionServer, CommStateMachine, SimpleActionClient, SimpleActionServer

from actionlib_msgs.msg import GoalStatus, GoalStatusArray

from long_term_deployment.srv import GetTaskFromID, GetTaskFromIDResponse

import rospy

from std_msgs.msg import Header


def get_task_from_gh(gh):
    task = None
    try:
        if gh is not None:
            task = gh.comm_state_machine.action_goal.goal.task
    except AttributeError as e:
        print('get_task_from_gh: {}'.format(e))
    return task


def get_id_from_gh(gh):
    try:
        goal_id = gh.comm_state_machine.action_goal.goal_id.id
    except AttributeError as e:
        print('get_id_from_gh: {}'.format(e))
        goal_id = None
    return goal_id


def null_task(task):
    ws_name_empty = task.workspace_name == ''
    pkg_name_empty = task.package_name == ''
    lf_name_empty = task.launchfile_name == ''
    if ws_name_empty and pkg_name_empty and lf_name_empty:
        return True
    else:
        return False


class SynchronizedActionClient(object):

    def __init__(self, action_namespace, action_spec):
        self.goals = []
        self.client = ActionClient(action_namespace, action_spec)
        self.status_topic = rospy.remap_name(action_namespace) + '/status'
        self.status_sub = rospy.Subscriber(
            self.status_topic,
            GoalStatusArray,
            self.add_missing_goals)
        # self.status_sub = rospy.Subscriber(
        #     rospy.remap_name(action_namespace) + '/goal',
        #     self.client.ActionGoal,
        #     self.add_new_goal)

        self.client.wait_for_server()
        rospy.wait_for_service(action_namespace + '/get_goal_from_id')
        self.goal_from_id = rospy.ServiceProxy(action_namespace + '/get_goal_from_id', GetTaskFromID)

    def send_goal(self, goal, transition_cb=None, feedback_cb=None):
        gh = self.client.send_goal(
            goal,
            transition_cb=transition_cb,
            feedback_cb=feedback_cb)

        self.goals.append(gh)

    def get_gh_from_task(self, task):
        for gh in self.goals:
            if task == get_task_from_gh(gh):
                return gh
        return None

    def add_missing_goals(self, status_array):
        tracked_goal_ids = [get_id_from_gh(gh) for gh in self.goals]
        for goal_status in status_array.status_list:
            if goal_status.status not in [2, 3, 4, 5, 8]:
                if goal_status.goal_id.id not in tracked_goal_ids:
                    goal = self.goal_from_id(goal_status.goal_id.id).goal
                    if not null_task(goal.task):
                        self.start_tracking_goal_id(goal_status.goal_id, goal)
                    else:
                        msg = 'Null goal received, not tracking {} with status {}'
                        rospy.logwarn(msg.format(goal_status.goal_id.id, goal_status))

    def add_new_goal(self, msg):
        self.start_tracking_goal_id(msg.goal_id, msg.goal)
        pass

    def start_tracking_goal_id(self, goal_id, goal):
        action_goal = self.client.manager.ActionGoal(
            header=Header(),
            goal_id=goal_id,
            goal=goal)

        csm = CommStateMachine(action_goal, None, None,
                               None, None)

        with self.client.manager.list_mutex:
            self.client.manager.statuses.append(weakref.ref(csm))

        self.goals.append(actionlib.ClientGoalHandle(csm))


class SynchronizedActionServer(object):

    def __init__(self, namespace, action_spec, goal_start_fn=None, goal_stop_fn=None):
        self.goals = {}
        self.goal_start_fn = goal_start_fn
        self.goal_stop_fn = goal_stop_fn
        self.goal_service = rospy.Service(
            namespace + '/get_goal_from_id',
            GetTaskFromID,
            self.task_id_cb)
        self.server = ActionServer(
            namespace,
            action_spec,
            self.receive_goal,
            self.stop_fn,
            auto_start=False)
        self.server.start()

    def task_id_cb(self, request):
        key = request.task_id
        if key in self.goals:
            rospy.loginfo('task_id_cb: successfully found task {}.'.format(key))
            gh = self.goals[key]
            return GetTaskFromIDResponse(gh.get_goal())
        else:
            rospy.logwarn('task_id_cb: could not find requested goal {}'.format(key))
            rospy.logwarn('task_id_cb: running goals: {}'.format(self.goals.keys()))
        return GetTaskFromIDResponse()

    def receive_goal(self, gh):
        self.goals[gh.get_goal_id().id] = gh
        print('Goals: {}'.format(self.goals.keys()))
        self.goal_start_fn(gh)

    def stop_fn(self, gh):
        self.goal_stop_fn(gh)


class SynchronizedSimpleActionClient(object):

    def __init__(self, action_namespace, action_spec):
        self.client = SimpleActionClient(action_namespace, action_spec)
        self.status_topic = rospy.remap_name(action_namespace) + '/status'
        # self.status_sub = rospy.Subscriber(
        #     self.status_topic,
        #     GoalStatusArray,
        #     self.add_missing_goal)
        # self.status_sub = rospy.Subscriber(
        #     rospy.remap_name(action_namespace) + '/goal',
        #     self.client.action_client.ActionGoal,
        #     self.add_new_goal)

        self.client.wait_for_server()
        rospy.wait_for_service(action_namespace + '/get_goal_from_id')
        self.goal_from_id = rospy.ServiceProxy(
            action_namespace + '/get_goal_from_id',
            GetTaskFromID)

    def send_goal(self, goal, done_cb=None, active_cb=None, feedback_cb=None):
        self.client.send_goal(
            goal,
            done_cb=done_cb,
            active_cb=active_cb,
            feedback_cb=feedback_cb)

    def get_state(self):
        if self.client.gh is None:
            return GoalStatus.LOST
        return self.client.get_state()

    def get_result(self):
        return self.client.get_result()

    def get_gh_from_task(self, task):
        if task == get_task_from_gh(self.client.gh):
            return self.client.gh
        return None

    def add_missing_goal(self, status_array):
        tracked_goal_id = get_id_from_gh(self.client.gh)
        for goal_status in status_array.status_list:
            if goal_status.goal_id.id != tracked_goal_id:
                goal = self.goal_from_id(goal_status.goal_id.id)
                self.start_tracking_goal_id(goal_status.goal_id, goal)

    def add_new_goal(self, msg):
        self.start_tracking_goal_id(msg.goal_id, msg.goal)

    def start_tracking_goal_id(self, goal_id, goal):
        action_goal = self.client.action_client.manager.ActionGoal(
            header=Header(),
            goal_id=goal_id,
            goal=goal)

        csm = CommStateMachine(action_goal, None, None,
                               None, None)

        # simple action client only tracks one goal at a time
        self.client.stop_tracking_goal()
        self.client.gh = actionlib.ClientGoalHandle(csm)

    def stop_tracking_goal(self):
        self.client.stop_tracking_goal()


class SynchronizedSimpleActionServer(object):

    def __init__(self, namespace, action_spec, execute_cb):
        self.goal_service = rospy.Service(
            namespace + '/get_goal_from_id',
            GetTaskFromID,
            self.task_id_cb)
        self.server = SimpleActionServer(
            namespace,
            action_spec,
            execute_cb,
            auto_start=False)
        self.server.start()

    def task_id_cb(self, request):
        idx = request.task_id
        current_goal = self.server.current_goal
        if idx == current_goal.goal_id.id:
            return GetTaskFromIDResponse(current_goal.get_goal())
        return GetTaskFromIDResponse()

    def is_preempt_requested(self):
        return self.server.is_preempt_requested()

    def set_preempted(self):
        return self.server.set_preempted()

    def set_succeeded(self, result):
        return self.server.set_succeeded(result)

    def publish_feedback(self, feedback):
        return self.server.publish_feedback(feedback)
