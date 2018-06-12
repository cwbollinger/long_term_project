#!/usr/bin/env python
import rospy
from tf import TransformListener, transformations
from std_msgs.msg import String, Int32
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from move_base_msgs.msg import MoveBaseGoal
from move_base_msgs.msg import MoveBaseAction
# from pr2_controllers_msgs.msg import PointHeadAction, PointHeadGoal, SingleJointPositionAction, SingleJointPositionGoal
from control_msgs.msg import PointHeadAction, PointHeadGoal, SingleJointPositionAction, SingleJointPositionGoal
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Quaternion
import actionlib
from actionlib_msgs.msg import GoalStatus
from moveit_msgs.msg import MoveItErrorCodes
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from image_geometry import PinholeCameraModel
import cv2
import math
import sets
import random
import datetime



class Node:
    def __init__(self, image_topic, camera_info_topic, camera_frame, torso_movement_topic, head_movement_topic):

        self.camera_frame = camera_frame
        self.bridge = CvBridge()
        self.tf = TransformListener()
        # rospy.Subscriber(robot_pose_topic, PoseWithCovarianceStamped, self.robot_pose_topic)
        rospy.Subscriber(image_topic, Image, self.image_callback)
        # rospy.Subscriber(points_topic, PointCloud2, self.pc_callback)
        rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)
        # rospy.Subscriber(published_point_num_topic, Int32, self.pub_point_num_callback)
        self.robot_pose = None
        self.img = None
        self.pc = None
        self.camera_info = None

        # base movement
        self.base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.base_client.wait_for_server()

        # # torso movement
        # self.torso_client = actionlib.SimpleActionClient(torso_movement_topic, SingleJointPositionAction)
        # self.torso_client.wait_for_server()

        # # head movement
        self.point_head_client = actionlib.SimpleActionClient(head_movement_topic, PointHeadAction)
        self.point_head_client.wait_for_server()

        rospy.loginfo("move group")
        #self.move_group = MoveGroupInterface("arm_with_torso", "base_link")
        rospy.loginfo("move group end")

    def robot_pose_callback(self, data):
        self.robot_pose = data

    def image_callback(self, data):
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def move_base_to(self, x, y, theta):
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        quat = transformations.quaternion_from_euler(0, 0, theta)
        orientation = Quaternion()
        orientation.x = quat[0]
        orientation.y = quat[1]
        orientation.z = quat[2]
        orientation.w = quat[3]
        goal.target_pose.pose.orientation = orientation

        # send goal
        self.base_client.send_goal(goal)


    # def move_torso(self, pose):
    #     goal = SingleJointPositionGoal()
    #     goal.position = pose
    #     self.torso_client.send_goal(goal)

    #def move_torso(self, pose):
    #    joint_names = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
    #    poses = [pose, 1.3192424769714355, 1.4000714648620605, -0.20049656002880095, 1.5290160491638183, -0.0004613047506046297, 1.660243449769287, -0.00012475593825578626]
    #    self.move_group.moveToJointPosition(joint_names, poses, wait=False)   # plan
    #    self.move_group.get_move_action().wait_for_result()
    #    result = self.move_group.get_move_action().get_result()
    #    return result


    def look_at(self, frame_id, x, y, z):
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame_id
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z

        goal.pointing_frame = "pointing_frame"
        goal.pointing_axis.x = 1
        goal.pointing_axis.y = 0
        goal.pointing_axis.z = 0

        # send goal
        self.point_head_client.send_goal(goal)


    # def pc_callback(self, data):
    #     self.pc = data

    def camera_info_callback(self, data):
        self.camera_info = data


    def get_img(self):
        return self.img

    def get_pc(self):
        return self.pc



def main():

    rospy.init_node('capture_data_cardbaord', anonymous=True)

    ar_tag_frame = rospy.get_param("~ar_tag_frame")
    image_filepath = rospy.get_param("~image_filepath")

    image_topic = "/head_camera/rgb/image_rect_color"
    camera_info_topic = "/head_camera/rgb/camera_info"
    map_frame = "map"
    robot_torso_frame = "torso_lift_link"
    camera_frame = "/head_camera_rgb_optical_frame"
    torso_movement_topic = "/torso_controller/follow_joint_trajectory"
    head_movement_topic = "/head_controller/point_head"

    node = Node(image_topic, camera_info_topic, camera_frame, torso_movement_topic, head_movement_topic)
    rospy.sleep(2.0)
    rospy.loginfo("Initialized")

    camera_model = PinholeCameraModel()
    while node.camera_info is None:     # wait for camera info
        continue
    camera_model.fromCameraInfo(node.camera_info)

    # look around for tag if robot can't see it
    offset = 0.2
    looking_points = [(1.125, 0., 0.65), (1.125, 0., 0.65 + offset), (1.125, 0. - offset, 0.65 + offset), (1.125, 0. - offset, 0.65), (1.125, 0. - offset, 0.65 - offset),
             (1.125, 0., 0.65 - offset), (1.125, 0. + offset, 0.65 - offset), (1.125, 0. + offset, 0.65), (1.125, 0. + offset, 0.65 + offset)]

    looking_point_index = 0
    while not(node.tf.frameExists(ar_tag_frame) and (looking_point_index < len(looking_points))):

        ps = PointStamped()
        ps.header.frame_id = robot_torso_frame
        ps.header.stamp = rospy.Time.now()
        ps.point.x = looking_points[looking_point_index][0]  # out
        ps.point.y = looking_points[looking_point_index][1]  # left
        ps.point.z = looking_points[looking_point_index][2]  # up

        rospy.loginfo("Looking for ar tag")
        node.look_at(robot_torso_frame, ps.point.x, ps.point.y, ps.point.z)
        result = node.point_head_client.wait_for_result()
        rospy.loginfo(result)

        looking_point_index += 1


    if node.tf.frameExists(ar_tag_frame):

        # fixed points relative to ar tag
        ps = PointStamped()
        ps.header.frame_id = ar_tag_frame
        #ps.header.frame_id = robot_torso_frame
        ps.header.stamp = node.tf.getLatestCommonTime(ar_tag_frame, map_frame)
        #ps.header.stamp = node.tf.getLatestCommonTime(robot_torso_frame, map_frame)

        if ar_tag_frame == "tag_1":
            ps.point.x = 0.150   # left
            ps.point.y = -0.850   # up
            ps.point.z = 0.150   # out
        elif ar_tag_frame == "tag_2":
            ps.point.x = -0.140  # left
            ps.point.y = -1.20 # up
            ps.point.z = 0.400   # out


        # transform point to map frame
        ps_new = node.tf.transformPoint(map_frame, ps)
        # ps_new = node.tf.transformPoint(robot_torso_frame, ps)

        # make robot look at object
        rospy.loginfo("Turning head toward cardboard")
        node.look_at("/map", ps_new.point.x, ps_new.point.y, ps_new.point.z)
        result = node.point_head_client.wait_for_result()
        rospy.loginfo(result)

        #if result.status == g_status.SUCCEEDED:
        if result == True:

            rospy.loginfo("Head turn succeeded")

            rospy.sleep(.1)


            # capture and save image
            img_cur = node.get_img()
            rospy.sleep(.1)
            if (img_cur is not None):

                rospy.loginfo("Capturing image")

                height, width, channels = img_cur.shape

                # save image along with pos annotations
                cur_time = str(datetime.datetime.now())
                image_file = image_filepath + "_" + cur_time + '.png'
                cv2.imwrite(image_file, img_cur)
                
                rospy.loginfo("Image saved")

    else:
        rospy.loginfo(ar_tag_frame)
        rospy.loginfo("Can't see ar tag")


if __name__ == "__main__":
    main()
