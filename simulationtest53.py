import pybullet as p
import pybullet_data
import time
import os
from utilities2 import *  # Original utilities
from utilities3 import *  # HDF5 recording utilities
from enhanced_camera import EnhancedWorkspaceCamera  
# Connect to PyBullet with error handling
try:
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    print("✅ Connected to PyBullet successfully!")
except Exception as e:
    print(f"❌ Failed to connect to PyBullet: {e}")
    exit(1)

p.resetSimulation()
p.setGravity(0, 0, -9.81)

# Load plane and robot
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
print("✅ Ground plane and robot loaded successfully!")

# Get controllable joints
controllable_joints = get_movable_joints(robot_id)
initialize_robot_joints(robot_id, controllable_joints)

# Load collection area
try:
    collection_area_id = p.loadURDF("short_floor.urdf", [-0.2, 0, 0.0])
    p.changeDynamics(collection_area_id, -1, mass=0)
    print("✅ Collection area loaded")
except Exception as e:
    collection_area_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.05]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.05], rgbaColor=[0.8, 0.7, 0.5, 1]),
        basePosition=[0, -0.5, 0]
    )
    print("✅ Created fallback collection platform")

# Create cups
cup_positions = [[0.6, 0, 0.3], [0.6, 0.2, 0.3], [0.6, -0.2, 0.3]]
cup_colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
cup_names = ["Red", "Green", "Blue"]
box_ids = []
color_names = []
create_cups(cup_positions, cup_colors, cup_names, box_ids, color_names)

# Store original positions for reset functionality
original_positions = {
    'cups': cup_positions.copy(),  # Store original cup positions
    'robot': [0, -0.785, 0, -2.356, 0, 1.571, 0.785],  # Robot home position
    'collection_area': [0, -0.5, 0.0]  # Collection area position
}

# Create systems
gripper = AutoGraspingSystem(robot_id, controllable_joints, box_ids, color_names)
trajectory_recorder = RobotTrajectoryRecorder(robot_id, controllable_joints, gripper)
main_camera = ProfessionalCameraSystem(gripper)

# Initialize workspace camera with SMALLER resolution for speed
# workspace_camera = WorkspaceCamera(
#     position=[1.0, -0.8, 0.7],
#     target=[0.0, 0.0, 0.1],
#     width=320,  # 🔥 REDUCED from 640 for speed
#     height=240  # 🔥 REDUCED from 480 for speed
# )
workspace_camera = EnhancedWorkspaceCamera(
    position=[1.0, -0.8, 0.7],
    target=[0.0, 0.0, 0.1],
    width=320,
    height=240,
    debug=True  # Enable debug messages
)

# 🆕 CREATE HDF5 RECORDING SYSTEM WITH DEMO DIRECTORY
# Create a timestamped demo directory
demo_timestamp = time.strftime("%Y%m%d_%H%M%S")
demo_dir = f"robot_demos_{demo_timestamp}"

hdf5_recorder = HDF5DemonstrationRecorder(
    filename=None,  # Will auto-generate filename
    demo_directory=demo_dir,  # 🆕 Custom demo directory
    image_shape=(240, 320),
    chunk_size=500
)

state_extractor = RobotStateExtractor(robot_id, controllable_joints)
action_extractor = ActionExtractor(robot_id, controllable_joints)

print("✅ HDF5 recording system initialized with DEMO DIRECTORY")

# Control state
control_mode = "teleop"
manual_target = [0, 0, 0, 0, 0, 0, 0]
target_position = np.array([0.6, 0, 0.3])

# Define control mappings
key_to_joint = {
    ord('q'): (0, +1), ord('a'): (0, -1),  # Joint 1
    ord('4'): (1, +1), ord('s'): (1, -1),  # Joint 2
    ord('e'): (2, +1), ord('d'): (2, -1),  # Joint 3
    ord('r'): (3, +1), ord('f'): (3, -1),  # Joint 4
    ord('t'): (4, +1), ord('g'): (4, -1),  # Joint 5
    ord('y'): (5, +1), ord('0'): (5, -1),  # Joint 6
    ord('u'): (6, +1), ord('j'): (6, -1),  # Joint 7
}

print("\n🤖 OPTIMIZED ROBOT CONTROL SYSTEM")
print("   1 = Teleop | 2 = Imitation | 3 = Autonomous")
print("   7 = Home | 8 = Ready | 9 = Short Floor")
print("   SPACE = CSV Recording | H = HDF5 Recording | G = Auto-Grasp")
print("   Q/W=J0, A/S=J1, Z/X=J2, E/R=J3, D/F=J4, C/V=J5, T/Y=J6")
print("   ESC = Exit")
print("\n⚡ PERFORMANCE: Reduced image size and recording frequency for smooth teleop")

frame_counter = 0
hdf5_skip_counter = 0

# 🔥 PERFORMANCE SETTINGS
HDF5_RECORD_EVERY_N_FRAMES = 70 # Record every 5th frame instead of every frame
MAX_RECORDING_TIME_PER_FRAME = 0.001  # Max 1ms for recording per frame

# Function to reset all items to their original positions
def reset_all_items_to_original():
    """Reset all items (cups, robot, collection area) to their original positions"""
    try:
        print("🔄 Resetting all items to original positions...")
        
        # 1. Reset robot to home position
        home_positions = original_positions['robot']
        for i, pos in enumerate(home_positions):
            if i < len(controllable_joints):
                p.resetJointState(robot_id, controllable_joints[i], pos)
        
        # 2. Reset cups to original positions
        for i, (box_id, original_pos) in enumerate(zip(box_ids, original_positions['cups'])):
            if box_id is not None:  # Check if box still exists
                # Reset position and orientation
                p.resetBasePositionAndOrientation(
                    box_id, 
                    original_pos, 
                    [0, 0, 0, 1]  # No rotation (quaternion)
                )
                # Reset velocity to zero
                p.resetBaseVelocity(box_id, [0, 0, 0], [0, 0, 0])
        
        # 3. Reset collection area (if moveable)
        if collection_area_id is not None:
            try:
                p.resetBasePositionAndOrientation(
                    collection_area_id,
                    original_positions['collection_area'],
                    [0, 0, 0, 1]
                )
            except:
                pass  # Collection area might be fixed
        
        # 4. Reset gripper state
        if hasattr(gripper, 'release'):
            gripper.release()
        elif hasattr(gripper, 'manual_release'):
            gripper.manual_release()
        
        # 5. Reset control variables
        global manual_target, target_position
        manual_target = list(original_positions['robot'])
        target_position = np.array([0.6, 0, 0.3])
        
        print("✅ All items reset successfully!")
        
    except Exception as e:
        print(f"❌ Error resetting items: {e}")

# Main simulation loop
while True:
    frame_counter += 1
    loop_start_time = time.time()
    
    # PRIORITY 1: Process keyboard events IMMEDIATELY
    keys = p.getKeyboardEvents()
    
    # Handle triggered keys
    for key, state in keys.items():
        if state & p.KEY_WAS_TRIGGERED:
            if key == 27:  # ESC
                print("🛑 Exit requested")
                if trajectory_recorder.recording:
                    trajectory_recorder.stop_recording()
                if hdf5_recorder.is_recording:
                    hdf5_recorder.stop_recording()
                p.disconnect()
                exit(0)
            
            elif key == ord('1'):
                control_mode = "teleop"
                print("🕹️ Teleop mode")
            elif key == ord('2'):
                control_mode = "imitation"
                print("🧠 Imitation mode")
            elif key == ord('3'):
                control_mode = "autonomous" 
                print("🤖 Autonomous mode")
            
            elif key == ord('7'):
                print("🏠 Moving to HOME...")
                move_to_preset(robot_id, controllable_joints, 'home')
            elif key == ord('8'):
                print("🚀 Moving to READY...")
                move_to_preset(robot_id, controllable_joints, 'ready')
            elif key == ord('9'):
                print("📦 Moving to SHORT FLOOR...")
                move_to_preset(robot_id, controllable_joints, 'short_floor')
            
            elif key == 32:  # SPACE
                if trajectory_recorder.recording:
                    filepath = trajectory_recorder.stop_recording()
                    print(f"⏹️ CSV stopped: {filepath}")
                else:
                    trajectory_recorder.start_recording()
                    print("🔴 CSV started")
            
            elif key == ord('h'):  # H
                if hdf5_recorder.is_recording:
                    success = hdf5_recorder.stop_recording()
                    print(f"⏹️ HDF5 stopped: {'Success' if success else 'Failed'}")
                else:
                    success = hdf5_recorder.start_recording()
                    if success:
                        print("🔴 HDF5 started - OPTIMIZED for smooth teleop!")
                    else:
                        print("❌ HDF5 failed to start")
            
            elif key == ord('z'):
                try:
                    # Most common attribute names in auto-grasping systems:
                    if hasattr(gripper, 'enabled'):
                        gripper.enabled = not gripper.enabled
                        status = "ON" if gripper.enabled else "OFF"
                    elif hasattr(gripper, 'auto_grasp'):
                        gripper.auto_grasp = not gripper.auto_grasp
                        status = "ON" if gripper.auto_grasp else "OFF"
                    elif hasattr(gripper, 'active'):
                        gripper.active = not gripper.active
                        status = "ON" if gripper.active else "OFF"
                    else:
                        status = "NOT AVAILABLE"
                    
                    print(f"🤏 Auto-grasping: {status}")
                except Exception as e:
                    print(f"❌ Auto-grasping toggle error: {e}")
            elif key== ord('x'):  # G
                try:
                   success=gripper.manual_release()
                   if not success:
                          print("❌ Gripper release failed")
                except Exception as e:
                    print(f"❌ Gripper open error: {e}")
            elif key == ord('i'):  # 🆕 Info about demos
                demo_info = hdf5_recorder.get_demo_info()
                print(f"\n📊 DEMO DIRECTORY INFO:")
                print(f"📁 Directory: {demo_info['demo_directory']}")
                print(f"💾 Current file: {os.path.basename(demo_info['current_file']) if demo_info['current_file'] else 'None'}")
                if demo_info['file_exists']:
                    print(f"📏 Current file size: {demo_info['file_size_mb']:.2f} MB")
                
                if demo_info['demo_files']:
                    print(f"📂 All demo files ({len(demo_info['demo_files'])}):")
                    for file_info in demo_info['demo_files']:
                        print(f"   • {file_info['filename']} ({file_info['size_mb']:.2f} MB)")
                else:
                    print("📂 No demo files found yet")
        
        # ADD THIS NEW CASE:
        elif key == ord('0'):  # Press '0' to reset all items
            reset_all_items_to_original()
        
    # PRIORITY 2: Handle teleop controls IMMEDIATELY (no delays)
    if control_mode == "teleop":
        for key, state in keys.items():
            if state & p.KEY_IS_DOWN:
                if key in key_to_joint:
                    joint_idx, direction = key_to_joint[key]
                    if joint_idx < len(controllable_joints):
                        joint_id = controllable_joints[joint_idx]
                        current_pos, _, _, _ = p.getJointState(robot_id, joint_id)
                        new_pos = current_pos + direction * 0.03  # 🔥 Slightly larger steps for responsiveness
                        
                        # Apply control IMMEDIATELY
                        p.setJointMotorControl2(
                            bodyUniqueId=robot_id,
                            jointIndex=joint_id,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=new_pos,
                            force=500,
                            maxVelocity=2.0  # 🔥 Faster movement
                        )
                        
                        if joint_idx < len(manual_target):
                            manual_target[joint_idx] = new_pos
    
    # PRIORITY 3: Update essential systems
    gripper.update_auto_grasping()
    main_camera.update_follow_camera()
    
    # PRIORITY 4: Record CSV (lightweight)
    if trajectory_recorder.recording:
        try:
            trajectory_recorder.record_frame()
        except Exception as e:
            print(f"❌ CSV error: {e}")
    
    # PRIORITY 5: Record HDF5 (OPTIMIZED - only occasionally)
    if hdf5_recorder.is_recording:
     hdf5_skip_counter += 1
    
    # Only record every Nth frame to reduce lag
    if hdf5_skip_counter >= HDF5_RECORD_EVERY_N_FRAMES:
        hdf5_skip_counter = 0
        
        # 🔥 CRITICAL FIX: Check PyBullet connection FIRST
        if not p.isConnected():
            print("❌ PyBullet connection lost - reconnecting...")
            try:
                p.connect(p.GUI)
                print("✅ Reconnected to PyBullet")
            except Exception as e:
                print(f"❌ Failed to reconnect: {e}")
                continue  # Skip this frame
        
        try:
            # Capture images with timeout protection
            rgb_img = workspace_camera.capture_image()
            
            # Only proceed if image is valid
            if rgb_img is not None and hasattr(rgb_img, 'shape'):
                # Get robot state (safely)
                try:
                    proprio_state = state_extractor.get_proprioceptive_state()
                except Exception as e:
                    print(f"⚠️ Proprioceptive error: {e}")
                    proprio_state = np.zeros(20, dtype=np.float32)  # Safe default
                
                # Get action (safely)
                try:
                    action = action_extractor.get_current_action()
                except Exception as e:
                    print(f"⚠️ Action error: {e}")
                    action = np.zeros(7, dtype=np.float32)  # Safe default
                
                # Handle tuple case for RGB image
                if isinstance(rgb_img, tuple):
                    rgb_img = rgb_img[0] if len(rgb_img) > 0 else None
                
                # 🔥 APPEND DATA - With critical safety checks
                success = hdf5_recorder.append_data(
                    rgb_image=rgb_img,
                    depth_image=None,  # No depth for now
                    proprio_data=proprio_state,
                    action=action,
                    future_traj=target_position
                )
                
                if success:
                    # Silent logging for smooth operation
                    if hdf5_recorder.timestep % 50 == 0:  # Report only every 50 frames
                        print(f"💾 HDF5 recording: {hdf5_recorder.timestep} frames")
                else:
                    print("❌ Failed to append data to HDF5")
            else:
                print("⚠️ Invalid camera image - skipping frame")
            
        except Exception as e:
            print(f"❌ HDF5 recording error: {e}")
            import traceback
            traceback.print_exc()
    # PRIORITY 6: Status (infrequent)
    if frame_counter % 300 == 0:  # Every ~1.25 seconds
        csv_status = "🔴" if trajectory_recorder.recording else "⚫"
        hdf5_status = "🔴" if hdf5_recorder.is_recording else "⚫"
        
        loop_time = time.time() - loop_start_time
        print(f"📊 {control_mode.upper()} | CSV:{csv_status} | HDF5:{hdf5_status} | Loop:{loop_time*1000:.1f}ms")
        
        if hdf5_recorder.is_recording:
            info = hdf5_recorder.get_status()
            print(f"    HDF5: {info['current_demo']} - {info['timestep']} steps")
    
    # PRIORITY 7: Step simulation
    p.stepSimulation()
    
    # 🔥 ADAPTIVE DELAY: Shorter delay when recording to maintain responsiveness
    if hdf5_recorder.is_recording:
        time.sleep(1./300.)  # Faster simulation when recording
    else:
        time.sleep(1./240.)  # Normal speed when not recording


