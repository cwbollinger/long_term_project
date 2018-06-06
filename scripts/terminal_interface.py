#!/usr/bin/env python

import sys,os
import curses

import rospy
from long_term_deployment.srv import *


def draw_menu(stdscr):
    global agents_proxy
    global tasks_proxy

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
    while k != ord('q') and not rospy.is_shutdown():
        tmp = stdscr.getch() # non blocking now
        if tmp != -1:
            k = tmp

        # Initialization
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        agents = agents_proxy().agents
        tasks = tasks_proxy().tasks

        # Declaration of strings
        keystr = "Last key pressed: {}".format(k)[:width-1]
        statusbarstr = "Press 'q' to exit | STATUS BAR"
        if k == 0:
            keystr = "No key press detected..."[:width-1]

        # Centering calculations
        start_x_keystr = int((width // 2) - (len(keystr) // 2) - len(keystr) % 2)
        start_y = int((height // 2) - 2)

        stdscr.addstr(2, 0, "-" * (width - 1))
        for i in range(1, height-1):
            stdscr.addstr(i, 15, '|')
            stdscr.addstr(i, 39, '|')
            stdscr.addstr(i, 59, "|")
       
        stdscr.addstr(1, 3, "Agent")
        stdscr.addstr(1, 22, "Active task")
        stdscr.addstr(1, 75,"Task Queue")

        for i, agent in enumerate(agents):
            stdscr.addstr(i+3, 62, agent.agent_name)
        for i, task in enumerate(tasks):
            stdscr.addstr(i+3, 62, task.data)

        # Rendering some text
        whstr = "Width: {}, Height: {}".format(width, height)
        stdscr.addstr(0, 0, whstr, curses.color_pair(1))

        # Render status bar
        stdscr.attron(curses.color_pair(3))
        stdscr.addstr(height-1, 0, statusbarstr)
        stdscr.addstr(height-1, len(statusbarstr), " " * (width - len(statusbarstr) - 1))
        stdscr.attroff(curses.color_pair(3))

        stdscr.addstr(start_y + 5, start_x_keystr, keystr)

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
    tasks_proxy = rospy.ServiceProxy('/task_server/get_queued_tasks', QueueTaskList)

    main()

#self.s1 = rospy.Service('{}/register_agent'.format(name), RegisterAgent, self.handle_register_agent)
