#!/usr/bin/env python
import roslib
roslib.load_manifest("visual_grasping")
import rospy
import utils
import openrave_bridge.pr2model
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import MarkerArray
import tf

import numpy as np

class VisualGrasper(object):
    def __init__(self, robot = None):
        if robot is None:
            self.robot = openrave_bridge.pr2model.PR2Robot()
        else:
            self.robot = robot
            
        self.gripper_pub = rospy.Publisher("~gripper_estimate", MarkerArray)
    
        self.tf_listener = tf.TransformListener()
    
    def gripper_from_points(self, p0, p1, frame_id = "/base_link"):
        """
        p0: [x,y,z]
        p1: [x,y,z]
        
        returns: a geometry_msgs PoseStamped
        """
        
        p1 =  np.asarray(p1)
        p0 = np.asarray(p0)
        dest = p0
        T = utils.make_orth_basis_z_ax(p1 - p0)

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
        
    
    def point_at_gripper(self, target,
                         desired_dist, 
                         gripper_x,
                         min_gripper_y, 
                         max_gripper_y,
                         n_attempts=1000,
                         constrain_y = lambda _ :True,
                         constrain_z = lambda _: True                          
                          ):
        """Given a target, move the gripper so that it's always
        pointing at its pose while keeping the desired_dist.
        """
        assert isinstance(target, PoseStamped)
        
        if target.header.frame_id != "/base_link":
            rospy.loginfo("Changing the frame from %s to %s", target.header.frame_id,
                          "/base_link")
            self.tf_listener.waitForTransform("/base_link", target.header.frame_id,
                                              rospy.Time.now(), rospy.Duration(1))
            target = self.tf_listener.transformPose("/base_link", target)
            
        tx = target.pose.position.x
        ty = target.pose.position.y
        tz = target.pose.position.z

        gripper_xyz = utils.valid_sphere_given_one_coord(desired_dist,
                                                         gripper_x,
                                                         min_gripper_y,
                                                         max_gripper_y,
                                                         (tx, ty, tz),
                                                         1, 
                                                         n_attempts,
                                                         constrain_y,
                                                         constrain_z
                                                         )
        if len(gripper_xyz) == 0:
            rospy.logerr("Could not find a valid solution")
            return False
        gripper_x, gripper_y, gripper_z = gripper_xyz[0]

        vec = (-gripper_x + tx,
               -gripper_y + ty,
               -gripper_z + tz,
               )
        
        rot_mat = utils.make_orth_basis(vec)
        
        M = np.identity(4)
        M[:3, :3] = rot_mat
        M[:3, 3] = (gripper_x, gripper_y, gripper_z)
        pos = utils.matrixToPose(M)
        
        ps = PoseStamped()
        ps.header.frame_id = "/base_link"
        ps.pose = pos
        return ps
    