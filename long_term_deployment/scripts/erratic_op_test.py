#! /usr/bin/env python
import rospy

from geometry_msgs.msg import Pose
from nav_msgs.srv import GetPlan

from long_term_deployment.msg import ConstrainedTask, Task
from long_term_deployment.srv import ScheduleTask, GetSchedule, AssignSchedule


if __name__ == '__main__':
    rospy.init_node('test_plan')
    schedule_task = rospy.ServiceProxy('/central_planner/schedule_task', ScheduleTask)

    get_schedule = rospy.ServiceProxy('/central_planner/get_schedule', GetSchedule)
    assign_schedule = rospy.ServiceProxy('/erratic/set_schedule', AssignSchedule)
    # use all test tasks for simplicity
    test_task = Task('', 'long_term_deployment', 'test_task', ['5'], False)

    curr_time = rospy.Time.now()
    while curr_time == 0.0:
        curr_time = rospy.Time.now()

    one_min = rospy.Duration(60)

    start_home = ConstrainedTask()
    start_home.location.position.x = 0.0
    start_home.location.position.y = 0.0
    start_home.task = test_task
    start_home.start_after = curr_time
    start_home.finish_before = curr_time + one_min
    start_home.estimated_duration = rospy.Duration(5.0)

    t1 = ConstrainedTask()
    t1.location.position.x = 5.0
    t1.location.position.y = 1.0
    t1.task = test_task
    t1.start_after = curr_time + one_min
    t1.finish_before = curr_time + 2 * one_min
    t1.estimated_duration = rospy.Duration(5.0)

    t2 = ConstrainedTask()
    t2.location.position.x = 1.0
    t2.location.position.y = 5.0
    t2.task = test_task
    t2.start_after = curr_time + 2 * one_min
    t2.finish_before = curr_time + 3 * one_min
    t2.estimated_duration = rospy.Duration(5.0)

    t3 = ConstrainedTask()
    t3.location.position.x = 5.0
    t3.location.position.y = 5.0
    t3.task = test_task
    t3.start_after = curr_time + 3 * one_min
    t3.finish_before = curr_time + 4 * one_min
    t3.estimated_duration = rospy.Duration(5.0)

    go_home = ConstrainedTask()
    go_home.location.position.x = 0.0
    go_home.location.position.y = 0.0
    go_home.task = test_task
    go_home.start_after = curr_time + 4 * one_min
    go_home.finish_before = curr_time + 5 * one_min
    go_home.estimated_duration = rospy.Duration(5.0)
    
    tasks = [start_home, t1, t2, t3, go_home]

    for task in tasks:
        schedule_task(task)

    result = get_schedule('erratic')
    print('Schedule received:')
    print(result)
    print(type(result))
    print('sending schedule in 5 seconds...')
    rospy.sleep(5)
    print('sending now!')
    assign_schedule(schedule=result.schedule)

    #r = rospy.Rate(1)
    #while not rospy.is_shutdown():
    #    r.sleep()
