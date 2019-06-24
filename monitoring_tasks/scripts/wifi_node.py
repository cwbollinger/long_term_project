#! /usr/bin/env python

import numpy as np

import rospy

from monitoring_tasks.msg import Connection
import subprocess

from tf import TransformListener
import tf2_ros


def fake_wifi_math(x, y):
    scale = 10.0
    sig_x = 70.0 * np.exp(-x / scale) * np.cos(scale * x)
    sig_y = 70.0 * np.exp(-y / scale) * np.cos(scale * y)
    sig = np.sqrt(sig_x**2 + sig_y**2)
    return int(round(sig))


class WifiNode:

    def __init__(self, *args):
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.wifi_pub = rospy.Publisher('~/wireless', Connection, queue_size=1)

    def main(self):
        r = rospy.Rate(1)
        if not rospy.has_param('/use_sim_time'):
            is_sim = False
        else:
            is_sim = rospy.get_param('/use_sim_time')

        if is_sim:
            wifi_func = self.get_fake_wifi
        else:
            wifi_func = self.get_wifi_status

        while not rospy.is_shutdown():
            conn = wifi_func()
            self.wifi_pub.publish(conn)
            r.sleep()

    def get_fake_wifi(self):
        position = None
        try:
            t = self.buffer.lookup_transform('map', 'base_link', rospy.Time())
            position = t.transform.translation
        except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print('Error finding transform...')
            return Connection()

        conn = Connection()
        conn.header.stamp = rospy.Time.now()
        conn.essid = 'SIM_AP'
        if position is not None:
            conn.quality = fake_wifi_math(position.x, position.y)

        return conn

    def get_wifi_status(self):
        net_status = subprocess.check_output(['sudo', 'iwconfig', 'wlan0'])
        fields = [x.strip() for x in net_status.split('\n')]
        conn = Connection()
        conn.header.stamp = rospy.Time.now()
        for field in fields:
            if 'ESSID' in field:
                conn.essid = field.split('"')[-2]
            elif 'Access' in field:
                conn.macaddr = field[-17:]
            elif 'Link Quality' in field:
                subfields = field.split()
                num = int(subfields[1][-5:-3])
                conn.quality = num  # number is out of 70
                conn.level = int(subfields[3][6:])
        return conn


if __name__ == '__main__':
    rospy.init_node('wifi_node')
    node = WifiNode()
    node.main()
