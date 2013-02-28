#!/usr/bin/env python
import roslib
roslib.load_manifest("visual_grasping")
import rospy
import utils
import openrave_bridge.pr2model
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import PointCloud2
import tf
import copy
import numpy as np
from pr2_control_utilities import IKUtilities

class VisualGrasper(object):
    def __init__(self, robot = None):
        if robot is None:
            self.robot = openrave_bridge.pr2model.PR2Robot()
        else:
            self.robot = robot
            
        self.gripper_pub = rospy.Publisher("~gripper_estimate", MarkerArray)    
        self.tf_listener = tf.TransformListener()
        self.left_ik = IKUtilities("left",
                                   tf_listener=self.tf_listener)
    
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
        fingertip = T.dot([-0.16,0,0,1])
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
                         n_poses = 100,
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
                                                         n_poses, 
                                                         n_attempts,
                                                         constrain_y,
                                                         constrain_z
                                                         )
        if len(gripper_xyz) == 0:
            rospy.logerr("Could not find a valid solution")
            return []
        all_poses = []
        for gripper_x, gripper_y, gripper_z in gripper_xyz:
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
            all_poses.append(ps)
        return all_poses
    
    def grasp_bowl(pose, p0, p1, x_angle, frame_id ="/base_link"):
        p1 =  np.asarray(p1)
        p0 = np.asarray(p0)
        if p0[2] > p1[2]:
            dest = p0
        else:
            dest = p1
        
        Q = utils.transformations.quaternion_from_euler(x_angle, 
                                                        np.pi/2, 
                                                        0)
        
        T = utils.transformations.quaternion_matrix(Q)
        T[:3, 3] = dest
        T[2,3] += 0.16
        
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose = utils.matrixToPose(T)
        return ps        
    
    def do_the_grasp(self, angle_in_degrees):
        bolt_points = rospy.wait_for_message("/bolt/vision/pcl_robot", 
                                             PointCloud2)
        all_points = utils.pointcloud2_to_xyz_array(bolt_points)
        p0 = all_points[0,:]
        p1 = all_points[2,:]
        angle = np.deg2rad(angle_in_degrees + 90)
        ps = self.grasp_bowl(p0, p1, angle)
        self.publish_gripper_pose(ps)
        approach = copy.deepcopy(ps)
        approach.pose.position.z += 0.2
        if ps.pose.position.y > 0:
            self.robot.move_left_arm(approach)
            self.robot.controller.open_left_gripper()
            self.robot.move_left_arm(ps)
            self.robot.controller.close_left_gripper()
        else:
            self.robot.move_right_arm(approach)
            self.robot.controller.open_right_gripper()
            self.robot.move_right_arm(ps)
            self.robot.controller.close_right_gripper()    
        
    def visible_trajectory(self, target, desired_dist):
        self.robot.controller.time_to_reach = 5.0
        ddist = (desired_dist-0.01)
        for gripper_x in np.linspace(target.pose.position.x - ddist, 
                                     target.pose.position.x + ddist):
        
            min_gripper_y = target.pose.position.y - ddist;
            max_gripper_y = target.pose.position.y + ddist;
            contrain_z = lambda z: z>target.pose.position.z+0.0
            all_gripper_ps = self.point_at_gripper(target, 
                                                   desired_dist, 
                                                   gripper_x, 
                                                   min_gripper_y,
                                                   max_gripper_y,
                                                   n_poses=20,
                                                   constrain_z=contrain_z)
            list_of_joints = []
            for gripper_ps in all_gripper_ps:
                M = utils.poseTomatrix(gripper_ps.pose)
                x,y,z = utils.transformations.euler_from_matrix(M)
                x += np.pi
                M2 = utils.transformations.euler_matrix(x,y,z)
                M[:3, :3] = M2[:3,:3]
                newpos = utils.matrixToPose(M)            
                gripper_ps.pose = newpos
                self.publish_gripper_pose(gripper_ps)

                curr_joint_angles = self.robot.controller.robot_state.left_arm_positions
                curr_joint_angles = np.array(curr_joint_angles)
                #joints, _ = self.left_ik.run_ik(gripper_ps,
                                                #curr_joint_angles,
                                                #"l_wrist_roll_link",
                                                #collision_aware=0)
                sols = self.robot.find_leftarm_ik(gripper_ps,
                                                            ignore_end_effector=False,
                                                            multiple_soluitions = True)
                if sols is not None and len(sols) != 0:
                    list_of_joints.extend(sols)
            list_of_joints = np.array(list_of_joints)
            if list_of_joints is not None and len(list_of_joints) != 0:
                print "Got %s solutions" % (list_of_joints.shape, )
                dists = np.sum((list_of_joints - curr_joint_angles)**2, 1)
                i = np.argmin(dists)
                joints = list_of_joints[i, :]
                
                self.robot.controller.time_to_reach = 1.5
                self.robot.controller.set_arm_state(joints, "left", wait=True)
        self.robot.controller.time_to_reach = 5.0
