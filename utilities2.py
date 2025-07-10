import csv
import datetime
import os
import time
import pybullet as p
import math
import numpy as np  # Add numpy for image array manipulation

# Add this to your existing utilities2.py file

class AutoGraspingSystem:
    def __init__(self, robot_id, controllable_joints, box_ids, color_names):
        self.robot_id = robot_id
        self.controllable_joints = controllable_joints
        self.box_ids = box_ids
        self.color_names = color_names
        self.current_grasp = None
        self.current_attachment = None
        self.auto_grasp_distance = 0.15  # Distance to auto-grasp
        self.auto_place_distance = 0.1  # Distance to auto-place
        self.collection_area_pos = [-1.2, 0, 0.0]  # Collection area position
        self.end_effector_link = 7
        self.last_check_time = 0
        self.check_interval = 0.05
        self.manual_release_requested = False
        self.gripper_closed = False
        self.is_near_collection_area = False
        
    def get_end_effector_pose(self):
        """Get end effector position and orientation"""
        try:
            link_state = p.getLinkState(self.robot_id, self.end_effector_link)
            return link_state[0], link_state[1]
        except:
            return p.getBasePositionAndOrientation(self.robot_id)
    
    def close_gripper(self):
        """Close gripper fingers"""
        # Find gripper joints
        gripper_joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if 'finger' in joint_name.lower() or 'gripper' in joint_name.lower():
                gripper_joints.append(i)
        
        if not gripper_joints:
            num_joints = p.getNumJoints(self.robot_id)
            gripper_joints = [num_joints - 2, num_joints - 1] if num_joints >= 2 else []
        
        for joint in gripper_joints:
            p.setJointMotorControl2(
                self.robot_id, joint, p.POSITION_CONTROL,
                targetPosition=0.02,
                force=50,
                maxVelocity=0.5
            )
        self.gripper_closed = True
        print("🤏 Gripper closed")
    
    def open_gripper(self):
        """Open gripper fingers"""
        # Find gripper joints
        gripper_joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if 'finger' in joint_name.lower() or 'gripper' in joint_name.lower():
                gripper_joints.append(i)
        
        if not gripper_joints:
            num_joints = p.getNumJoints(self.robot_id)
            gripper_joints = [num_joints - 2, num_joints - 1] if num_joints >= 2 else []
        
        for joint in gripper_joints:
            p.setJointMotorControl2(
                self.robot_id, joint, p.POSITION_CONTROL,
                targetPosition=0.04,
                force=50,
                maxVelocity=0.5
            )
        self.gripper_closed = False
        print("🖐️ Gripper opened")
    
    def find_nearest_object(self):
        """Find nearest grippable object"""
        ee_pos, _ = self.get_end_effector_pose()
        
        min_distance = float('inf')
        nearest_object = None
        
        for obj_id in self.box_ids:
            if self.current_grasp and self.current_grasp.body == obj_id:
                continue
            
            try:
                obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
                distance = math.sqrt(sum([(a-b)**2 for a, b in zip(ee_pos, obj_pos)]))
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_object = obj_id
            except:
                continue
        
        return nearest_object, min_distance
    
    def check_near_collection_area(self):
        """Check if gripper is near collection area"""
        ee_pos, _ = self.get_end_effector_pose()
        collection_pos = self.collection_area_pos
        
        # Calculate distance to collection area
        distance = math.sqrt(
            (ee_pos[0] - collection_pos[0])**2 + 
            (ee_pos[1] - collection_pos[1])**2 + 
            (ee_pos[2] - (collection_pos[2] + 0.3))**2  # Add height offset
        )
        
        was_near = self.is_near_collection_area
        self.is_near_collection_area = distance <= self.auto_place_distance
        
        # Debug info when entering/leaving collection area
        if self.is_near_collection_area and not was_near:
            print(f"🎯 Entered collection area! Distance: {distance:.2f}m")
        elif was_near and not self.is_near_collection_area:
            print(f"↩️ Left collection area. Distance: {distance:.2f}m")
        
        return self.is_near_collection_area, distance
    
    def generate_grasp(self, obj_id):
        """Generate a simple grasp for the given object"""
        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
        
        # Create constraint-based attachment
        constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.end_effector_link,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.05],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1]
        )
        
        p.changeConstraint(constraint, maxForce=300)
        
        return SimpleGrasp(obj_id, constraint)
    
    def auto_grip_object(self, obj_id):
        """Automatically grip an object"""
        try:
            self.close_gripper()
            time.sleep(0.1)
            
            grasp = self.generate_grasp(obj_id)
            
            if grasp and grasp.constraint:
                self.current_grasp = grasp
                
                try:
                    obj_name = self.color_names[self.box_ids.index(obj_id)]
                except (ValueError, IndexError):
                    obj_name = f"Cup_{obj_id}"
                
                print(f"🤖 AUTO-GRIPPED: {obj_name}")
                return True
            else:
                print("❌ Failed to create grasp constraint")
                self.open_gripper()
                return False
        except Exception as e:
            print(f"Error in auto_grip_object: {e}")
            return False
    
    def auto_place_object(self):
        """Automatically place object in collection area"""
        if self.current_grasp is not None:
            try:
                # Get object name before releasing
                try:
                    obj_name = self.color_names[self.box_ids.index(self.current_grasp.body)]
                except (ValueError, IndexError):
                    obj_name = f"Cup_{self.current_grasp.body}"
                
                # Remove constraint
                p.removeConstraint(self.current_grasp.constraint)
                
                # Open gripper
                self.open_gripper()
                
                # Clear current grasp
                self.current_grasp = None
                
                print(f"🏆 AUTO-PLACED: {obj_name} in collection area!")
                return True
                
            except Exception as e:
                print(f"Error in auto_place_object: {e}")
                return False
        return False
    
    def manual_release(self):
        """Manually release the currently held object"""
        if self.current_grasp is not None:
            self.manual_release_requested = True
            try:
                obj_name = self.color_names[self.box_ids.index(self.current_grasp.body)]
            except (ValueError, IndexError):
                obj_name = f"Cup_{self.current_grasp.body}"
            
            p.removeConstraint(self.current_grasp.constraint)
            self.open_gripper()
            self.current_grasp = None
            self.manual_release_requested = False
            
            print(f"🤲 MANUAL RELEASE: {obj_name}")
            return True
        else:
            print("🤲 No object to release")
            return False
    
    def update_auto_grasping(self):
        """Update auto-grasping and auto-placement system"""
        current_time = time.time()
        
        # Throttle checking for better performance
        if current_time - self.last_check_time < self.check_interval:
            return
        
        self.last_check_time = current_time
        
        try:
            # Check if near collection area
            near_collection, collection_distance = self.check_near_collection_area()
            
            # Auto-placement logic
            if self.current_grasp is not None and near_collection:
                print(f"🎯 Near collection area with cup! Auto-placing...")
                self.auto_place_object()
                return
            
            # Auto-grasping logic (only if not holding anything)
            if self.current_grasp is None:
                nearest_object, distance = self.find_nearest_object()
                
                if nearest_object is not None and distance <= self.auto_grasp_distance:
                    self.auto_grip_object(nearest_object)
                    
        except Exception as e:
            print(f"Error in update_auto_grasping: {e}")
    
    def get_status(self):
        """Get gripper status"""
        gripper_status = "CLOSED" if self.gripper_closed else "OPEN"
        collection_status = " 🎯 NEAR COLLECTION" if self.is_near_collection_area else ""
        
        if self.current_grasp:
            try:
                obj_name = self.color_names[self.box_ids.index(self.current_grasp.body)]
            except (ValueError, IndexError):
                obj_name = f"Cup_{self.current_grasp.body}"
            return f"GRIPPER {gripper_status} - Holding {obj_name}{collection_status}"
        return f"GRIPPER {gripper_status}{collection_status}"
    
    def get_current_conf(self):
        """Get current robot configuration"""
        return f"Config_{self.robot_id}"

class SimpleGrasp:
    """Simple grasp representation"""
    def __init__(self, body, constraint):
        self.body = body
        self.constraint = constraint

# Add the WorkspaceCamera class after existing classes
class WorkspaceCamera:
   
    def __init__(self, 
                 position=None, 
                 target=None, 
                 up_vector=None,
                 width=640, 
                 height=480, 
                 fov=60.0,
                 near_val=0.1,
                 far_val=10.0,
                 image_dir='robot_camera_feed'):
      
        # Default side view if not specified
        self.position = position if position else [1.5, 1.5, 0.8]
        self.target = target if target else [0.0, 0.0, 0.3]
        self.up_vector = up_vector if up_vector else [0, 0, 1]
        
        # Camera parameters
        self.width = width
        self.height = height
        self.fov = fov
        self.near_val = near_val
        self.far_val = far_val
        
        # For saving images
        self.image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)
        
        # Keep track of last captured image paths
        self.last_rgb_path = None
        self.last_depth_path = None
        self.last_segmentation_path = None
        
        # View presets
        self.view_presets = {
            'side': {
                'position': [1.5, 1.5, 0.8],
                'target': [0.0, 0.0, 0.3],
                'up_vector': [0, 0, 1]
            },
            'top': {
                'position': [0.0, 0.0, 1.8],
                'target': [0.0, 0.0, 0.0],
                'up_vector': [0, 1, 0]
            },
            'collection_area': {
                'position': [-1.0, 1.0, 0.8],
                'target': [-1.2, 0.0, 0.1],
                'up_vector': [0, 0, 1]
            },
            'workspace': {
                'position': [0.8, 1.2, 0.8],
                'target': [0.6, 0.0, 0.3],
                'up_vector': [0, 0, 1]
            }
        }
        
        print(f"📷 Workspace camera initialized: {width}x{height}, FOV {fov}°")
    
    def set_view_preset(self, preset_name):
        """Set camera to a predefined view preset"""
        if preset_name in self.view_presets:
            preset = self.view_presets[preset_name]
            self.position = preset['position']
            self.target = preset['target']
            self.up_vector = preset['up_vector']
            print(f"📷 Camera set to preset: {preset_name}")
            return True
        else:
            print(f"❌ Unknown camera preset: {preset_name}")
            return False
    
    def set_custom_view(self, position, target, up_vector=None):
        """Set a custom camera view"""
        self.position = position
        self.target = target
        self.up_vector = up_vector if up_vector else [0, 0, 1]
        print(f"📷 Camera set to custom position at {position}")
        return True
    
    def get_view_matrix(self):
        """Get the camera view matrix"""
        return p.computeViewMatrix(
            cameraEyePosition=self.position,
            cameraTargetPosition=self.target,
            cameraUpVector=self.up_vector
        )
    
    def get_projection_matrix(self):
        """Get the camera projection matrix"""
        aspect = self.width / self.height
        return p.computeProjectionMatrixFOV(
            fov=self.fov, 
            aspect=aspect,
            nearVal=self.near_val, 
            farVal=self.far_val
        )
    
    def capture_image(self, include_depth=False, include_segmentation=False):
       
        # Get camera matrices
        view_matrix = self.get_view_matrix()
        proj_matrix = self.get_projection_matrix()
        
        # Capture image
        img_flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if include_segmentation else 0
        w, h, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            flags=img_flags,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to numpy arrays
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(h, w, 4)
        
        if include_depth:
            depth_array = np.array(depth_img).reshape(h, w)
        else:
            depth_array = None
            
        if include_segmentation:
            seg_array = np.array(seg_img).reshape(h, w)
        else:
            seg_array = None
            
        return rgb_array, depth_array, seg_array
    
    def save_image(self, filename=None):
        """Save captured image to file"""
        if self.last_image is None:
            print("❌ No image captured yet")
            return False
        
        if filename is None:
            import time
            filename = f"workspace_{int(time.time())}.png"
        
        # Create directory if it doesn't exist
        import os
        os.makedirs("camera_images", exist_ok=True)
        
        # Use OpenCV to save the image instead of p.savePngFile
        import cv2
        rgb_path = os.path.join("camera_images", filename)
        
        # Convert RGB to BGR for OpenCV
        bgr_img = cv2.cvtColor(self.last_image, cv2.COLOR_RGB2BGR)
        
        # Save the image
        success = cv2.imwrite(rgb_path, bgr_img)
        
        if success:
            print(f"📸 Image saved to {rgb_path}")
        else:
            print("❌ Failed to save image")
        
        return success
    
    def preview_in_simulator(self):
        """Set the main PyBullet view to match this camera's viewpoint"""
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,  # This value doesn't matter, it's overridden
            cameraYaw=0,         # This value doesn't matter, it's overridden
            cameraPitch=0,       # This value doesn't matter, it's overridden
            cameraTargetPosition=self.target
        )
        
        # Use the computeViewMatrixFromYawPitchRoll to get the desired view
        # Calculate the direction vector from position to target
        direction = [self.target[0] - self.position[0],
                     self.target[1] - self.position[1],
                     self.target[2] - self.position[2]]
        
        # Calculate distance
        distance = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        
        # Calculate yaw angle (around z-axis)
        yaw = math.degrees(math.atan2(direction[1], direction[0]))
        
        # Calculate pitch angle (up/down)
        pitch = -math.degrees(math.asin(direction[2] / distance))
        
        # Reset camera using calculated values
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=self.target
        )
        
        print(f"📷 Preview: pos={self.position}, target={self.target}, distance={distance:.2f}, yaw={yaw:.1f}°, pitch={pitch:.1f}°")

# **REST OF YOUR EXISTING FUNCTIONS REMAIN THE SAME**
def get_movable_joints(body):
    """Get movable joints of a body"""
    movable_joints = []
    for i in range(p.getNumJoints(body)):
        joint_info = p.getJointInfo(body, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            movable_joints.append(i)
    return movable_joints

def initialize_robot_joints(robot_id, joint_indices):
    """Initialize robot joints to home position"""
    home_positions = [0.0] * len(joint_indices)
    for i, joint_index in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_index, home_positions[i])
    print(f"Robot joints initialized to home position: {home_positions}")
    return home_positions

def create_cups(cup_positions, cup_colors, cup_names, box_ids, color_names):
    """Create lightweight cups"""
    print("Creating cups...")
    for i, (pos, color) in enumerate(zip(cup_positions, cup_colors)):
        try:
            cup_id = p.loadURDF("cube.urdf", pos, globalScaling=0.1)
            p.changeVisualShape(cup_id, -1, rgbaColor=color)
            
            p.changeDynamics(cup_id, -1, 
                            mass=1,
                            lateralFriction=1.2,
                            rollingFriction=0.15,
                            spinningFriction=0.1,
                            restitution=0.2,
                            linearDamping=0.1,
                            angularDamping=0.1)
            
            box_ids.append(cup_id)
            color_names.append(f"{cup_names[i]} Cup")
            print(f"✅ {cup_names[i]} cup created at {pos}")
            
        except Exception as e:
            print(f"❌ Error creating cup for {cup_names[i]}: {e}")
    
    return box_ids

class ProfessionalCameraSystem:
    def __init__(self, gripper_system):
        self.gripper_system = gripper_system
        self.current_view = 0
        self.views = [
            [2.5, 30, -20, [0.0, 0, 0.4]],      # Overview
            [2.0, 90, -15, [0.0, 0, 0.4]],      # Side view
            [1.5, 0, -10, [0.6, 0, 0.6]],       # Cups area
            [3.0, 30, -60, [0.0, 0, 0.3]],      # Top view
            [1.8, 150, -10, [-1.2, 0, 0.4]],    # Collection focus (updated position)
            [1.2, 0, 0, [0, 0, 0.8]],            # End effector
        ]
        self.view_names = [
            "Overview", "Side View", "Cups Area", 
            "Top View", "Collection Focus", "End Effector"
        ]
        self.set_view(0)
    
    def set_view(self, view_index):
        """Set camera to specific view"""
        if 0 <= view_index < len(self.views):
            self.current_view = view_index
            view = self.views[view_index]
            
            if view_index == 5:  # End effector follow view
                ee_pos, _ = self.gripper_system.get_end_effector_pose()
                target = [ee_pos[0], ee_pos[1], ee_pos[2]]
            else:
                target = view[3]
            
            p.resetDebugVisualizerCamera(
                cameraDistance=view[0],
                cameraYaw=view[1], 
                cameraPitch=view[2],
                cameraTargetPosition=target
            )
            
            print(f"📷 Camera: {self.view_names[view_index]}")
    
    def cycle_view(self):
        """Cycle to next camera view"""
        self.current_view = (self.current_view + 1) % len(self.views)
        self.set_view(self.current_view)
    
    def update_follow_camera(self):
        """Update camera if in follow mode"""
        if self.current_view == 5:  # End effector follow mode
            try:
                ee_pos, _ = self.gripper_system.get_end_effector_pose()
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.2,
                    cameraYaw=0,
                    cameraPitch=0,
                    cameraTargetPosition=[ee_pos[0], ee_pos[1], ee_pos[2]]
                )
            except Exception as e:
                pass

def move_to_preset(robot_id, controllable_joints, preset_name):
    """Move robot to preset position"""
    preset_positions = {
        'home': [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],
        'ready': [0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.8],
        'short_floor': [-1.2, -0.5, 0.0, -1.8, 0.0, 1.3, 0.8],  # Position towards collection area
    }
    
    if preset_name in preset_positions:
        target_angles = preset_positions[preset_name]
        
        for i, angle in enumerate(target_angles):
            if i < len(controllable_joints):
                joint_idx = controllable_joints[i]
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id, jointIndex=joint_idx, controlMode=p.POSITION_CONTROL,
                    targetPosition=angle, force=1500, maxVelocity=0.8
                )
        
        print(f"🎯 Moving robot to {preset_name.upper()} position")
        return True
    return False

def get_comprehensive_status(gripper_system, camera_system, controllable_joints, cup_ids, spawner=None):
    """Get detailed system status"""
    try:
        ee_pos, _ = gripper_system.get_end_effector_pose()
        
        # Count cups in collection area
        collected_count = 0
        collection_pos = [-1.2, 0, 0.0]
        
        for cup_id in cup_ids:
            try:
                cup_pos, _ = p.getBasePositionAndOrientation(cup_id)
                distance_to_collection = math.sqrt(
                    (cup_pos[0] - collection_pos[0])**2 + 
                    (cup_pos[1] - collection_pos[1])**2
                )
                if distance_to_collection < 0.4 and cup_pos[2] < 0.4:
                    collected_count += 1
            except:
                continue
        
        status = f"""
📊 CUP MANIPULATION SYSTEM:
   🤖 End Effector: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]
   🔷 Gripper: {gripper_system.get_status()}
   📷 Camera: {camera_system.view_names[camera_system.current_view]}
   ☕ Cups in Collection: {collected_count}/{len(cup_ids)}
   🎯 Collection Area: [-1.2, 0, 0.0]
   🎯 Active Joints: {len(controllable_joints)}"""
        
        return status
        
    except Exception as e:
        return f"📊 STATUS ERROR: {e}"

# **KEYBOARD CONTROL MAPPING**
KEY_TO_DIRECTION = {
    ord('q'): (0, +1), ord('a'): (0, -1),  # Joint 1
    ord('4'): (1, +1), ord('s'): (1, -1),  # Joint 2
    ord('e'): (2, +1), ord('d'): (2, -1),  # Joint 3
    ord('r'): (3, +1), ord('f'): (3, -1),  # Joint 4
    ord('t'): (4, +1), ord('g'): (4, -1),  # Joint 5
    ord('y'): (5, +1), ord('h'): (5, -1),  # Joint 6
    ord('u'): (6, +1), ord('j'): (6, -1),  # Joint 7
}

def get_key_to_direction_mapping():
    """Get keyboard to joint direction mapping"""
    return KEY_TO_DIRECTION

class RobotTrajectoryRecorder:
    def __init__(self, robot_id, controllable_joints, gripper_system=None):
        self.robot_id = robot_id
        self.controllable_joints = controllable_joints
        self.gripper_system = gripper_system
        self.trajectory_data = []
        self.recording = False
        self.start_time = 0
        self.sample_rate = 10  # Hz
        self.last_record_time = 0
        
    def start_recording(self, filename=None):
        """Start trajectory recording"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.csv"
        
        self.filename = filename
        self.trajectory_data = []
        self.recording = True
        self.start_time = time.time()
        self.last_record_time = 0
        
        # CSV Headers
        headers = ['time', 'step']
        
        # Joint data
        for i, joint_idx in enumerate(self.controllable_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            joint_name = joint_info[1].decode('utf-8')
            headers.extend([f'{joint_name}_pos', f'{joint_name}_vel', f'{joint_name}_torque'])
        
        # Add end effector
        headers.extend(['ee_x', 'ee_y', 'ee_z', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw'])
        
        self.trajectory_data.append(headers)
        print(f"🎬 Recording started: {filename}")
        
    def record_step(self):
        """Record current robot state"""
        if not self.recording:
            return
            
        current_time = time.time()
        if current_time - self.last_record_time < (1.0 / self.sample_rate):
            return
            
        elapsed_time = current_time - self.start_time
        step = len(self.trajectory_data) - 1  # -1 for header
        
        row = [f"{elapsed_time:.3f}", step]
        
        # Record joint states
        for joint_idx in self.controllable_joints:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            position = joint_state[0]
            velocity = joint_state[1]
            
            # Get applied torque
            joint_info = p.getJointState(self.robot_id, joint_idx)
            applied_torque = joint_info[3]  # Applied joint motor torque
            
            row.extend([f"{position:.6f}", f"{velocity:.6f}", f"{applied_torque:.6f}"])
        
        # Record end effector
        if self.gripper_system:
            ee_pos, ee_orn = self.gripper_system.get_end_effector_pose()
        else:
            link_state = p.getLinkState(self.robot_id, 7)  # Assuming link 7 is end effector
            ee_pos, ee_orn = link_state[0], link_state[1]
        
        for pos in ee_pos:
            row.append(f"{pos:.6f}")
        for quat in ee_orn:
            row.append(f"{quat:.6f}")
        
        self.trajectory_data.append(row)
        self.last_record_time = current_time
        
    def stop_recording(self):
        """Stop recording and save CSV"""
        if not self.recording:
            return None
            
        self.recording = False
        
        # Create directory
        os.makedirs("robot_trajectories", exist_ok=True)
        filepath = os.path.join("robot_trajectories", self.filename)
        
        # Save CSV
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.trajectory_data)
            
            duration = time.time() - self.start_time
            points = len(self.trajectory_data) - 1
            
            print(f"💾 Trajectory saved: {filepath}")
            print(f"📊 Duration: {duration:.1f}s, Points: {points}, Rate: {points/duration:.1f} Hz")
            
            return filepath
        except Exception as e:
            print(f"❌ Error saving: {e}")
            return None