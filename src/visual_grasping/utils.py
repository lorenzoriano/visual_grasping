import euclid
import copy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Pose
from tf import transformations

def matrix4ToPose(matrix):
    """Converts a Matrix 4 to a Pose message by extracting the translation and
    rotation specified by the Matrix.

    Parameters:
    matrix: a Matrix4 (euclid.py)
    """
    pos = Pose()
    pos.position.x = matrix.d
    pos.position.y = matrix.h
    pos.position.z = matrix.l
    
    q = matrix.get_quaternion()
    pos.orientation.x = q.x
    pos.orientation.y = q.y
    pos.orientation.z = q.z
    pos.orientation.w = q.w

    return pos

def poseTomatrix4(pose):
    """Converts a Pose message to a matrix 4.
    """
    assert isinstance(pose, Pose)
    
    mat = euclid.Matrix4()
    mat.translate(pose.position.x,
                      pose.position.y,
                      pose.position.z, )        
    rot = euclid.Quaternion(pose.orientation.w,
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z, )
    mat *= rot.get_matrix()
    
    return mat

def makeGripperMarker(pose, angle=0.541, color=None, scale=1.0):
    """
    Creates an MarkerArray with the PR2 gripper shape. 
    Parameters:
    pose: the position of the marker, a Pose instance
    angle: the aperture angle of the gripper (default=0.541)
    color: (r,g,b,a) tuple or None (default) if using the material colors
    scale: the scale of the gripper, default is 1.0

    Returns:
    The new gripper MarkerArray
    """
    assert isinstance(pose, PoseStamped)
    pose_T = poseTomatrix4(pose.pose)

    gripper_marker = MarkerArray()

    T1 = euclid.Matrix4()
    T2 = euclid.Matrix4()

    T1.translate(0.07691, 0.01, 0.)
    T1.rotate_axis(angle, euclid.Vector3(0,0,1))
    T2.translate(0.09137, 0.00495, 0.)
    T1.rotate_axis(-angle, euclid.Vector3(0,0,1))

    T_proximal = T1.copy()
    T_distal = T1 * T2

    mesh = Marker()
    mesh.header.frame_id = pose.header.frame_id
    mesh.type = Marker.MESH_RESOURCE
    mesh.scale.x = scale
    mesh.scale.y = scale
    mesh.scale.z = scale
    
    if color is not None:
        mesh.color.r = color[0]
        mesh.color.g = color[1]
        mesh.color.b = color[2]
        mesh.color.a = color[3]
        mesh.mesh_use_embedded_materials = False
    else:
        mesh.mesh_use_embedded_materials = True

    mesh.mesh_resource = "package://pr2_description/meshes/gripper_v0/gripper_palm.dae"
    mesh.pose = matrix4ToPose(pose_T)
    #mesh.pose.orientation.w = 1
    gripper_marker.markers.append(copy.deepcopy(mesh))
    
    mesh.mesh_resource = "package://pr2_description/meshes/gripper_v0/l_finger.dae"
    mesh.pose = matrix4ToPose(pose_T * T_proximal)
    gripper_marker.markers.append(copy.deepcopy(mesh))

    mesh.mesh_resource = "package://pr2_description/meshes/gripper_v0/l_finger_tip.dae"
    mesh.pose = matrix4ToPose(pose_T * T_distal)
    gripper_marker.markers.append(copy.deepcopy(mesh))

    T1 = euclid.Matrix4()
    T2 = euclid.Matrix4()

    T1.translate(0.07691, -0.01, 0.)
    T1.rotate_axis(np.pi, euclid.Vector3(1,0,0))
    T1.rotate_axis(angle, euclid.Vector3(0,0,1))
    T2.translate(0.09137, 0.00495, 0.)
    T1.rotate_axis(-angle, euclid.Vector3(0,0,1))

    T_proximal = T1.copy()
    T_distal = T1 * T2

    mesh.mesh_resource = "package://pr2_description/meshes/gripper_v0/l_finger.dae"
    mesh.pose = matrix4ToPose(pose_T * T_proximal)
    gripper_marker.markers.append(copy.deepcopy(mesh))

    mesh.mesh_resource = "package://pr2_description/meshes/gripper_v0/l_finger_tip.dae"
    mesh.pose = matrix4ToPose(pose_T * T_distal)
    gripper_marker.markers.append(copy.deepcopy(mesh))

    for i, marker in enumerate(gripper_marker.markers):
        marker.id = i

    return gripper_marker

def make_orth_basis(z_ax):
    """
    orthogonal basis from a given z axis
    John Schulman magic code.
    """
    z_ax = np.asarray(z_ax)
    z_ax = z_ax / np.linalg.norm(z_ax)
    
    if np.allclose(z_ax, [0, 0, 1]) or np.allclose(z_ax, [0, 0, -1]):
        raise Exception("singular values!")
    
    x_ax = np.array([0, 0, -1.])
    x_ax -= z_ax * x_ax.dot(z_ax)
    y_ax = np.cross(z_ax, x_ax)
    
    T = np.c_[x_ax, y_ax, z_ax]
    #building a 4x4 matrix
    M = np.eye(4)
    M[:3, :3] = T
    return M