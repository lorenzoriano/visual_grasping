#!/usr/bin/env python
import roslib
roslib.load_manifest("visual_grasping")
import rospy
import utils
import openrave_bridge.pr2model
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import MarkerArray

import numpy as np

class VisualGrasper(object):
    def __init__(self, robot = None):
        if robot is None:
            self.robot = openrave_bridge.pr2model.PR2Robot()
        else:
            self.robot = robot
            
        self.gripper_pub = rospy.Publisher("~gripper_estimate", MarkerArray)
    
    def gripper_from_points(self, p0, p1, frame_id = "/base_link"):
        """
        p0: [x,y,z]
        p1: [x,y,z]
        
        returns: a geometry_msgs PoseStamped
        """
        
        p1 =  np.asarray(p1)
        p0 = np.asarray(p0)
        dest = p0
        T = utils.make_orth_basis(p1 - p0)

        #moving the gripper backward so that the fingers touch the point
        T[:3,3] = dest
        fingertip = T.dot([-0.18,0,0,1])
        T[:,3] = fingertip

        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose = utils.matrixToPose(T)
        return ps
    
    def publish_gripper_pose(self, pose):
        assert isinstance(pose, PoseStamped)
        gripper = utils.makeGripperMarker(pose)
        self.gripper_pub.publish(gripper)