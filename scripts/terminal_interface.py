#!/usr/bin/env python

import sys,os
import curses

import rospy
from long_term_deployment.srv import *


def draw_menu(stdscr):
    global agents_proxy
    global queued_tasks_proxy
    global active_tasks_proxy
    global queue_task_proxy

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
        queued_tasks = [t.data for t in queued_tasks_proxy().tasks]
        active_agents = [a.data for a in active_tasks_proxy().agents]
        active_tasks = [t.data for t in active_tasks_proxy().tasks]

        if num == 0:
            queue_task_proxy('long_term_ws','long_term_deployment','test_task.launch')
            num = None
        elif num == 1:
            queue_task_proxy('long_term_ws','tmp_tasks','cardboard_capture_1.launch')
            num = None
        elif num == 2:
            queue_task_proxy('long_term_ws','tmp_tasks','cardboard_capture_2.launch')
            num = None

        # Declaration of strings
        statusbarstr = "Press 'q' to exit | STATUS BAR"

        stdscr.addstr(2, 0, "-" * (width - 1))
        stdscr.addstr(int(height//2), 0, "-" * (width - 1))
        stdscr.addstr(int(height//2)+1, 0, 'Press "0" to queue "test_task"')
        stdscr.addstr(int(height//2)+2, 0, 'Press "1" to queue "cardboard_capture_1"')
        stdscr.addstr(int(height//2)+3, 0, 'Press "2" to queue "cardboard_capture_2"')
        for i in range(1, int(height//2)):
            stdscr.addstr(i, 15, '|')
            stdscr.addstr(i, 39, '|')
            stdscr.addstr(i, 59, "|")
       
        stdscr.addstr(1, 3, "Agent")
        stdscr.addstr(1, 22, "Active task")
        stdscr.addstr(1, 75,"Task Queue")

        for i, agent in enumerate(agents):
            stdscr.addstr(i+3, 2, agent)
            if agent in active_agents:
                idx = active_agents.index(agent)
                stdscr.addstr(i+3, 22, active_tasks[idx])
            else:
                stdscr.addstr(i+3, 22, "Inactive")

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
    rospy.wait_for_service('/task_server/get_active_tasks')
    active_tasks_proxy = rospy.ServiceProxy('/task_server/get_active_tasks', ActiveTaskList)
    rospy.wait_for_service('/task_server/queue_task')
    queue_task_proxy = rospy.ServiceProxy('/task_server/queue_task', QueueTask)

    main()

#self.s1 = rospy.Service('{}/register_agent'.format(name), RegisterAgent, self.handle_register_agent)
