#!/usr/bin/env python
import roslib
roslib.load_manifest("visual_grasping")
import rospy
import visual_grasper
import openrave_bridge.env_operations
from sensor_msgs.msg import PointCloud2, PointCloud
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
import tabletop_actions
import sys

robot = openrave_bridge.pr2model.PR2Robot()
detector = tabletop_actions.object_detector.GenericDetector()
detector.detect();
if detector.last_detection_msg is None:
    rospy.logerr("No table detected, tilt the head down!")
    sys.exit(0)

openrave_bridge.env_operations.add_table(
    detector.last_detection_msg, robot.env)
vg = visual_grasper.VisualGrasper(robot)
vg.work_on_bolt_pointcloud()
    

