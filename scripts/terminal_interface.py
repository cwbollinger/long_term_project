#!/usr/bin/env python

import sys
import os
import curses

import rospy
from long_term_deployment.msg import AgentDescription, Task
from long_term_deployment.srv import GetRegisteredAgents, QueueTask, QueueTaskList, AgentStatusList

def draw_menu(stdscr):
    global agents_proxy
    global queued_tasks_proxy
    global active_tasks_proxy
    global queue_active_task_proxy
    global start_continuous_task_proxy

    curses.raw()
    curses.cbreak()
    stdscr.nodelay(True)

    k = 0

    # Clear and refresh the screen for a blank canvas
    stdscr.clear()
    stdscr.refresh()

    # Start colors in curses
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)

    # Loop where k is the last character pressed
    r = rospy.Rate(10)
    num = None
    while k != ord('q') and not rospy.is_shutdown():
        tmp = stdscr.getch() # non blocking now
        if tmp != -1:
            k = tmp
            # convert to numeric
            num = k-48

        # Initialization
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        agents = [a.agent_name for a in agents_proxy().agents]
        queued_tasks = ['{}/{}'.format(t.package_name, t.launchfile_name) for t in queued_tasks_proxy().tasks]
        agent_tasks = {}
        active_tasks = []
        for a in active_tasks_proxy().agent_statuses:
            agent_tasks[a.agent.agent_name] = {
                    'active_task': a.active_task.launchfile_name,
                    'background_tasks': [t.launchfile_name for t in a.background_tasks],
            }

        if num == 0:
            queue_active_task_proxy(Task('','long_term_deployment','test_task', ['5'], False), AgentDescription('turtlebot', 'turtlebot'))
        elif num == 1:
            queue_active_task_proxy(Task('','cardboard_detection_task','cardboard_capture', ['1']))
        elif num == 2:
            queue_active_task_proxy(Task('','cardboard_detection_task','cardboard_capture', ['2']))
        elif num == 3:
            start_continuous_task_proxy(Task('','navigation_tasks','navigate_on_map', ['maze'], True), AgentDescription('turtlebot', 'turtlebot'))
        elif num == 4:
            stop_continuous_task_proxy(Task('','navigation_tasks','navigate_on_map', ['maze'], False), AgentDescription('turtlebot', 'turtlebot'))
        elif num == 5:
            start_continuous_task_proxy(Task('','navigation_tasks','build_new_map', ['new_maze'], False), AgentDescription('turtlebot', 'turtlebot'))
        elif num == 6:
            stop_continuous_task_proxy(Task('','navigation_tasks','build_new_map', ['new_maze'], False), AgentDescription('turtlebot', 'turtlebot'))
        elif num == 7:
            start_continuous_task_proxy(Task('','navigation_tasks','explore_map', [], True), AgentDescription('turtlebot', 'turtlebot'))
        elif num == 8:
            stop_continuous_task_proxy(Task('','navigation_tasks','explore_map', [], False), AgentDescription('turtlebot', 'turtlebot'))
        elif num == 9:
            queue_active_task_proxy(Task('','navigation_tasks','go_to_pose', ['5.0', '5.0', '0.0'], False), AgentDescription('turtlebot', 'turtlebot'))

        num = None

        # Declaration of strings
        statusbarstr = "Press 'q' to exit | STATUS BAR"

        stdscr.addstr(2, 0, "-" * (width - 1))
        stdscr.addstr(int(height//2), 0, "-" * (width - 1))
        stdscr.addstr(int(height//2)+1, 0, 'Press "0" to queue "test_task"')
        stdscr.addstr(int(height//2)+2, 0, 'Press "1" to queue "cardboard_capture_1"')
        stdscr.addstr(int(height//2)+3, 0, 'Press "2" to queue "cardboard_capture_2"')
        stdscr.addstr(int(height//2)+4, 0, 'Press "3" to start navigation on turtlebot agent')
        stdscr.addstr(int(height//2)+5, 0, 'Press "4" to stop navigation on turtlebot agent')
        stdscr.addstr(int(height//2)+6, 0, 'Press "5" to start building a map on turtlebot agent')
        stdscr.addstr(int(height//2)+7, 0, 'Press "6" to stop building a map on turtlebot agent')
        stdscr.addstr(int(height//2)+8, 0, 'Press "7" to start frontier exploration on turtlebot agent')
        stdscr.addstr(int(height//2)+9, 0, 'Press "8" to stop frontier exploration on turtlebot agent')
        stdscr.addstr(int(height//2)+10, 0, 'Press "9" to go to 5,5 on map')

        for i in range(1, int(height//2)):
            stdscr.addstr(i, 15, '|')
            stdscr.addstr(i, 39, '|')
            stdscr.addstr(i, 59, "|")

        stdscr.addstr(1, 3, "Agent")
        stdscr.addstr(1, 22, "Active task")
        stdscr.addstr(1, 41, "Continuous tasks")
        stdscr.addstr(1, 68,"Task Queue")

        for i, agent in enumerate(agents):
            stdscr.addstr(i+3, 2, agent)
            
            active_task = agent_tasks[agent]['active_task']
            if active_task is None:
                stdscr.addstr(i+3, 22, "Inactive")
            else:
                stdscr.addstr(i+3, 22, active_task)

            stdscr.addstr(i+3, 41, str(agent_tasks[agent]['background_tasks']))

        for i, task in enumerate(queued_tasks):
            stdscr.addstr(i+3, 62, task)

        # Rendering some text
        whstr = "Width: {}, Height: {}".format(width, height)
        stdscr.addstr(0, 0, whstr, curses.color_pair(1))

        # Render status bar
        stdscr.attron(curses.color_pair(3))
        stdscr.addstr(height-1, 0, statusbarstr)
        stdscr.addstr(height-1, len(statusbarstr), " " * (width - len(statusbarstr) - 1))
        stdscr.attroff(curses.color_pair(3))


        # Refresh the screen
        stdscr.refresh()
        r.sleep()

def main():
    curses.wrapper(draw_menu)

if __name__ == "__main__":
    rospy.init_node('server_terminal')
    rospy.wait_for_service('/task_server/get_agents')
    agents_proxy = rospy.ServiceProxy('/task_server/get_agents', GetRegisteredAgents)
    rospy.wait_for_service('/task_server/get_queued_tasks')
    queued_tasks_proxy = rospy.ServiceProxy('/task_server/get_queued_tasks', QueueTaskList)
    rospy.wait_for_service('/task_server/get_agents_status')
    active_tasks_proxy = rospy.ServiceProxy('/task_server/get_agents_status', AgentStatusList)
    rospy.wait_for_service('/task_server/queue_task')
    queue_active_task_proxy = rospy.ServiceProxy('/task_server/queue_task', QueueTask)
    rospy.wait_for_service('/task_server/start_continuous_task')
    start_continuous_task_proxy = rospy.ServiceProxy('/task_server/start_continuous_task', QueueTask)
    rospy.wait_for_service('/task_server/stop_continuous_task')
    stop_continuous_task_proxy = rospy.ServiceProxy('/task_server/stop_continuous_task', QueueTask)

    main()

#self.s1 = rospy.Service('{}/register_agent'.format(name), RegisterAgent, self.handle_register_agent)
