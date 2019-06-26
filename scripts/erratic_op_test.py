#! /usr/bin/env python
import rospy

from geometry_msgs.msg import Pose
from nav_msgs.srv import GetPlan

from long_term_deployment.msg import ConstrainedTask, Task
from long_term_deployment.srv import ScheduleTask, GetSchedule


if __name__ == '__main__':
    rospy.init_node('test_plan')
    schedule_task = rospy.ServiceProxy('/central_planner/schedule_task', ScheduleTask)

    get_schedule = rospy.ServiceProxy('/central_planner/get_schedule', GetSchedule)
    # use all test tasks for simplicity
    test_task = Task('', 'long_term_deployment', 'test_task', ['5'], False)

    curr_time = rospy.Time.now()
    while curr_time == 0.0:
        curr_time = rospy.Time.now()

    one_min = rospy.Duration(60)
    two_min = rospy.Duration(120)
    three_min = rospy.Duration(180)
    four_min = rospy.Duration(240)

    t1 = ConstrainedTask()
    t1.location = Pose()
    t1.location.position.x = 1.0
    t1.location.position.y = 5.0
    t1.task = test_task
    t1.start_after = curr_time + one_min
    t1.finish_before = curr_time + two_min
    t1.estimated_duration = rospy.Duration(5.0)

    t2 = ConstrainedTask()
    t2.location = Pose()
    t2.location.position.x = 2.0
    t2.location.position.y = 7.0
    t2.task = test_task
    t2.start_after = curr_time + two_min
    t2.finish_before = curr_time + three_min
    t2.estimated_duration = rospy.Duration(5.0)

    t3 = ConstrainedTask()
    t3.location = Pose()
    t3.location.position.x = 5.0
    t3.location.position.y = 5.0
    t3.task = test_task
    t3.start_after = curr_time + three_min
    t3.finish_before = curr_time + four_min
    t3.estimated_duration = rospy.Duration(5.0)
    tasks = [t1, t2, t3]

    for task in tasks:
        schedule_task(task)

    result = get_schedule('erratic')
    print('Schedule received:')
    print(result)

    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        r.sleep()
