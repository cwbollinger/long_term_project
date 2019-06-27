#! /usr/bin/env python
import threading

import rospy

def main(stop_event, args, client_params): # doesn't handle preemption, but whatever
    start_time = float(args[0]) if len(args) > 0 else 5.0
    sleepytime = start_time
    while sleepytime > 0.0:
        rospy.loginfo('{}'.format(sleepytime))
        rospy.sleep(1.0 if sleepytime > 1.0 else sleepytime)
        sleepytime -= 1.0
    return {'status':'success', 'duration':'{} seconds'.format(start_time)}

if __name__ == "__main__":
    rospy.init_node('test_node') # can't do this in main or it flips out
    main(threading.Event(), [])
