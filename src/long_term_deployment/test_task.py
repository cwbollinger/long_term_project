import yaml
import threading

import rospy

def main(stop_event, args): # doesn't handle preemption, but whatever
    sleepytime = float(args[0]) if len(args) > 0 else 5
    rospy.sleep(sleepytime)
    return yaml.dump({'status':'done sleepin', 'duration':'{} seconds'.format(sleepytime)})

if __name__ == "__main__":
    rospy.init_node('test_node') # can't do this in main or it flips out
    main(threading.Event())
