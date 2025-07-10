import h5py
import numpy as np
import pybullet as p
import cv2
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import threading
from pathlib import Path

class HDF5DemonstrationRecorder:
    """
    Records robot demonstrations to HDF5 format with streaming capability.
    Stores images, proprioceptive data, and actions in a structured format.
    """
    
    def __init__(self, 
                 filename: str = None,
                 demo_directory: str = "demos",
                 image_shape: Tuple[int, int] = (480, 640),
                 max_demos: int = 100,
                 chunk_size: int = 1000):
        """
        Initialize HDF5 recorder
        
        Args:
            filename: HDF5 file path. If None, auto-generates with timestamp
            demo_directory: Directory to save demonstrations in
            image_shape: (Height, Width) of captured images
            max_demos: Maximum number of demonstrations to store
            chunk_size: HDF5 chunk size for efficient I/O
        """
        self.image_shape = image_shape
        self.chunk_size = chunk_size
        self.max_demos = max_demos
        self.demo_directory = demo_directory
        
        # CREATE DEMO DIRECTORY STRUCTURE
        self._create_demo_directory()
        
        # Generate filename if not provided
        if filename is None:
            filename = f"demonstrations_11.hdf5"
        
        # SAVE IN DEMO DIRECTORY
        self.filename = os.path.join(self.demo_directory, filename)
        self.file = None
        self.current_demo = None
        self.demo_count = 0
        self.timestep = 0
        self.is_recording = False
        
        # Datasets for current demo
        self.datasets = {}
        
        print(f"🎬 HDF5 Recorder initialized")
        print(f"📁 Demo directory: {os.path.abspath(self.demo_directory)}")
        print(f"💾 File will be saved as: {os.path.abspath(self.filename)}")
    
    def _create_demo_directory(self):
        """Create demo directory structure"""
        try:
            # Create main demo directory
            Path(self.demo_directory).mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for organization
            subdirs = ["raw_demos", "processed_demos", "metadata"]
            for subdir in subdirs:
                Path(os.path.join(self.demo_directory, subdir)).mkdir(exist_ok=True)
            
            print(f"📁 Created demo directory structure:")
            print(f"   📂 {self.demo_directory}/")
            print(f"   ├── 📂 raw_demos/")
            print(f"   ├── 📂 processed_demos/")
            print(f"   └── 📂 metadata/")
            
        except Exception as e:
            print(f"⚠️ Could not create demo directory: {e}")
            print(f"📁 Using current directory instead")
            self.demo_directory = "."
    
    def get_demo_info(self) -> dict:
        """Get information about saved demos"""
        info = {
            'demo_directory': os.path.abspath(self.demo_directory),
            'current_file': os.path.abspath(self.filename) if hasattr(self, 'filename') else None,
            'file_exists': os.path.exists(self.filename) if hasattr(self, 'filename') else False,
            'file_size_mb': 0,
            'demo_files': []
        }
        
        # Get file size if exists
        if info['file_exists']:
            info['file_size_mb'] = os.path.getsize(self.filename) / (1024 * 1024)
        
        # List all HDF5 files in demo directory
        if os.path.exists(self.demo_directory):
            for file in os.listdir(self.demo_directory):
                if file.endswith('.hdf5'):
                    filepath = os.path.join(self.demo_directory, file)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    info['demo_files'].append({
                        'filename': file,
                        'path': filepath,
                        'size_mb': size_mb,
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                    })
        
        return info

    def start_recording(self, demo_name: str = None) -> bool:
        """
        Start recording a new demonstration
        
        Args:
            demo_name: Optional name for the demo. If None, uses demo_{count}
            
        Returns:
            bool: True if recording started successfully
        """
        if self.is_recording:
            print("⚠️ Already recording! Stop current recording first.")
            return False
        
        try:
            # Open/create HDF5 file
            self.file = h5py.File(self.filename, 'a')
            
            # Create data group if it doesn't exist
            if 'data' not in self.file:
                self.file.create_group('data')
            
            # Generate demo name
            if demo_name is None:
                self.demo_count = len([k for k in self.file['data'].keys() if k.startswith('demo_')]) + 1
                demo_name = f"demo_{self.demo_count}"
            
            self.current_demo = demo_name
            
            # Create demo group
            if demo_name in self.file['data']:
                print(f"⚠️ Demo {demo_name} already exists, overwriting...")
                del self.file['data'][demo_name]
            
            demo_group = self.file['data'].create_group(demo_name)
            
            # Create observation group
            obs_group = demo_group.create_group('obs')
            
            # Pre-define datasets with unlimited time dimension
            self.datasets = {
                # RGB images: (T, 3, H, W)
                'rgb': obs_group.create_dataset(
                    'rgb',
                    shape=(0, 3, self.image_shape[0], self.image_shape[1]),
                    maxshape=(None, 3, self.image_shape[0], self.image_shape[1]),
                    dtype=np.uint8,
                    chunks=(self.chunk_size, 3, self.image_shape[0], self.image_shape[1]),
                    compression='gzip',
                    compression_opts=1
                ),
                
                # Depth images: (T, 1, H, W)
                'depth': obs_group.create_dataset(
                    'depth',
                    shape=(0, 1, self.image_shape[0], self.image_shape[1]),
                    maxshape=(None, 1, self.image_shape[0], self.image_shape[1]),
                    dtype=np.float32,
                    chunks=(self.chunk_size, 1, self.image_shape[0], self.image_shape[1]),
                    compression='gzip',
                    compression_opts=1
                ),
                
                # Proprioceptive data: (T, P) where P depends on robot
                'proprio': obs_group.create_dataset(
                    'proprio',
                    shape=(0, 20),
                    maxshape=(None, 20),
                    dtype=np.float32,
                    chunks=(self.chunk_size, 20),
                    compression='gzip',
                    compression_opts=1
                ),
                
                # Actions: (T, A) where A is action dimension
                'actions': demo_group.create_dataset(
                    'actions',
                    shape=(0, 7),
                    maxshape=(None, 7),
                    dtype=np.float32,
                    chunks=(self.chunk_size, 7),
                    compression='gzip',
                    compression_opts=1
                ),
                
                # Future trajectory targets: (T, 3) for XYZ positions
                'traj_future': demo_group.create_dataset(
                    'traj/future',
                    shape=(0, 3),
                    maxshape=(None, 3),
                    dtype=np.float32,
                    chunks=(self.chunk_size, 3),
                    compression='gzip',
                    compression_opts=1
                )
            }
            
            self.timestep = 0
            self.is_recording = True
            
            print(f"🔴 Started recording demo: {demo_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            if self.file:
                self.file.close()
                self.file = None
            return False

    def append_data(self, 
                   rgb_image: np.ndarray,
                   depth_image: np.ndarray = None,
                   proprio_data: np.ndarray = None,
                   action: np.ndarray = None,
                   future_traj: np.ndarray = None) -> bool:
        """
        Append data for current timestep
        """
        print(f"🔄 append_data called - timestep: {self.timestep}, recording: {self.is_recording}")
        
        if not self.is_recording:
            print("⚠️ Not recording! Call start_recording() first.")
            return False
        
        try:
            # Process RGB image
            if rgb_image is not None:
                # Convert tuple or other formats to numpy array
                if isinstance(rgb_image, tuple):
                    print("⚠️ RGB image is tuple, extracting first element")
                    rgb_image = rgb_image[0] if len(rgb_image) > 0 else None
                
                if rgb_image is not None:
                    # Ensure it's a numpy array
                    if not isinstance(rgb_image, np.ndarray):
                        rgb_image = np.array(rgb_image, dtype=np.uint8)
                    
                    # Handle different shapes
                    if len(rgb_image.shape) == 3:
                        if rgb_image.shape[2] == 4:  # RGBA, remove alpha
                            rgb_image = rgb_image[:, :, :3]
                        
                        # Ensure correct data type
                        if rgb_image.dtype != np.uint8:
                            rgb_image = rgb_image.astype(np.uint8)
                        
                        # Resize if needed
                        if rgb_image.shape[:2] != self.image_shape:
                            rgb_image = cv2.resize(rgb_image, (self.image_shape[1], self.image_shape[0]))
                        
                        # Convert from (H, W, 3) to (3, H, W)
                        rgb_chw = np.transpose(rgb_image, (2, 0, 1))
                    else:
                        print(f"⚠️ Unexpected RGB shape: {rgb_image.shape}, creating dummy")
                        rgb_chw = np.zeros((3, self.image_shape[0], self.image_shape[1]), dtype=np.uint8)
                else:
                    print("⚠️ RGB image is None, creating dummy")
                    rgb_chw = np.zeros((3, self.image_shape[0], self.image_shape[1]), dtype=np.uint8)
                
                # Resize dataset and append
                self.datasets['rgb'].resize((self.timestep + 1, 3, self.image_shape[0], self.image_shape[1]))
                self.datasets['rgb'][self.timestep] = rgb_chw
            
            # Process depth image
            if depth_image is not None:
                if isinstance(depth_image, tuple):
                    depth_image = depth_image[0] if len(depth_image) > 0 else None
                
                if depth_image is not None:
                    if not isinstance(depth_image, np.ndarray):
                        depth_image = np.array(depth_image, dtype=np.float32)
                    
                    if len(depth_image.shape) == 2:
                        depth_chw = depth_image[np.newaxis, :, :]
                    elif len(depth_image.shape) == 3:
                        if depth_image.shape[2] == 1:
                            depth_chw = np.transpose(depth_image, (2, 0, 1))
                        else:
                            depth_chw = depth_image[0:1, :, :]
                    else:
                        print(f"⚠️ Unexpected depth shape: {depth_image.shape}")
                        depth_chw = np.zeros((1, self.image_shape[0], self.image_shape[1]), dtype=np.float32)
                else:
                    depth_chw = np.zeros((1, self.image_shape[0], self.image_shape[1]), dtype=np.float32)
                
                self.datasets['depth'].resize((self.timestep + 1, 1, self.image_shape[0], self.image_shape[1]))
                self.datasets['depth'][self.timestep] = depth_chw
            
            # Process proprioceptive data
            if proprio_data is not None:
                if isinstance(proprio_data, tuple):
                    proprio_data = np.array(proprio_data, dtype=np.float32)
                elif not isinstance(proprio_data, np.ndarray):
                    proprio_data = np.array(proprio_data, dtype=np.float32)
                
                # Pad or truncate to match dataset size
                proprio_padded = np.zeros(20, dtype=np.float32)
                min_len = min(len(proprio_data.flatten()), 20)
                proprio_data_flat = proprio_data.flatten()
                proprio_padded[:min_len] = proprio_data_flat[:min_len]
                
                self.datasets['proprio'].resize((self.timestep + 1, 20))
                self.datasets['proprio'][self.timestep] = proprio_padded
            
            # Process actions
            if action is not None:
                if isinstance(action, tuple):
                    action = np.array(action, dtype=np.float32)
                elif not isinstance(action, np.ndarray):
                    action = np.array(action, dtype=np.float32)
                
                # Pad or truncate to match dataset size
                action_padded = np.zeros(7, dtype=np.float32)
                min_len = min(len(action.flatten()), 7)
                action_flat = action.flatten()
                action_padded[:min_len] = action_flat[:min_len]
                
                self.datasets['actions'].resize((self.timestep + 1, 7))
                self.datasets['actions'][self.timestep] = action_padded
            
            # Process future trajectory
            if future_traj is not None:
                if isinstance(future_traj, tuple):
                    future_traj = np.array(future_traj, dtype=np.float32)
                elif not isinstance(future_traj, np.ndarray):
                    future_traj = np.array(future_traj, dtype=np.float32)
                
                traj_padded = np.zeros(3, dtype=np.float32)
                min_len = min(len(future_traj.flatten()), 3)
                traj_flat = future_traj.flatten()
                traj_padded[:min_len] = traj_flat[:min_len]
                
                self.datasets['traj_future'].resize((self.timestep + 1, 3))
                self.datasets['traj_future'][self.timestep] = traj_padded
            
            # INCREMENT TIMESTEP - CRITICAL!
            self.timestep += 1
            
            # Flush data periodically
            if self.timestep % 100 == 0:
                self.file.flush()
                print(f"💾 HDF5 data flushed at timestep {self.timestep}")
            
            # RETURN SUCCESS - CRITICAL!
            return True
            
        except Exception as e:
            print(f"❌ Failed to append data at timestep {self.timestep}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop_recording(self) -> bool:
        """
        Stop current recording and close file
        
        Returns:
            bool: True if stopped successfully
        """
        if not self.is_recording:
            print("⚠️ Not currently recording!")
            return False
        
        try:
            # Add metadata
            if self.current_demo and self.file:
                demo_group = self.file['data'][self.current_demo]
                demo_group.attrs['total_timesteps'] = self.timestep
                demo_group.attrs['recorded_at'] = "datetime.now().isoformat()"
                demo_group.attrs['image_shape'] = self.image_shape
            
            # Close file
            if self.file:
                self.file.close()
                self.file = None
            
            print(f"⏹️ Stopped recording {self.current_demo} ({self.timestep} timesteps)")
            print(f"💾 Saved to: {self.filename}")
            
            self.is_recording = False
            self.current_demo = None
            self.datasets = {}
            self.timestep = 0
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to stop recording: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current recording status"""
        return {
            'is_recording': self.is_recording,
            'current_demo': self.current_demo,
            'timestep': self.timestep,
            'filename': self.filename,
            'demo_count': self.demo_count
        }

class EnhancedWorkspaceCamera:
    """
    Enhanced camera system for capturing both RGB and depth images
    """
    
    def __init__(self, 
                 position: List[float] = [1.0, 0, 0.7],
                 target: List[float] = [0.0, 0.0, 0.1],
                 width: int = 640,
                 height: int = 480,
                 fov: float = 60,
                 near: float = 0.1,
                 far: float = 100):
        
        self.position = position
        self.target = target
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        
        # Pre-compute matrices
        self.update_matrices()
    
    def update_matrices(self):
        """Update view and projection matrices"""
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.position,
            cameraTargetPosition=self.target,
            cameraUpVector=[0, 0, 1]
        )
        
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far
        )
    
    def capture_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture both RGB and depth images
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (rgb_image, depth_image)
                rgb_image: (H, W, 3) uint8 array
                depth_image: (H, W) float32 array
        """
        try:
            # Capture image
            width, height, rgb_img, depth_img, _ = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=self.view_matrix,
                projectionMatrix=self.proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Process RGB
            rgb_array = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))
            rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
            
            # Process depth
            depth_array = np.array(depth_img, dtype=np.float32).reshape((height, width))
            
            # Convert depth from [0,1] to actual distances
            depth_array = self.far * self.near / (self.far - (self.far - self.near) * depth_array)
            
            return rgb_array, depth_array
            
        except Exception as e:
            print(f"❌ Camera capture failed: {e}")
            # Return dummy data
            rgb_dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            depth_dummy = np.ones((self.height, self.width), dtype=np.float32)
            return rgb_dummy, depth_dummy
    
    def capture_image(self) -> np.ndarray:
        """Capture only RGB image for backward compatibility"""
        rgb, _ = self.capture_rgbd()
        return rgb

class RobotStateExtractor:
    """
    Extract proprioceptive state from robot
    """
    
    def __init__(self, robot_id: int, controllable_joints: List[int]):
        self.robot_id = robot_id
        self.controllable_joints = controllable_joints
    
    def get_proprioceptive_state(self) -> np.ndarray:
        """
        Get current proprioceptive state
        
        Returns:
            np.ndarray: State vector containing joint positions, velocities, 
                       end-effector pose, etc.
        """
        try:
            state_vector = []
            
            # Joint positions and velocities
            for joint_idx in self.controllable_joints:
                pos, vel, _, _ = p.getJointState(self.robot_id, joint_idx)
                state_vector.extend([pos, vel])
            
            # End-effector pose (position + orientation)
            ee_link_id = 7  # Assuming panda_link8
            ee_state = p.getLinkState(self.robot_id, ee_link_id)
            ee_pos = ee_state[0]  # Position
            ee_orn = ee_state[1]  # Orientation (quaternion)
            
            state_vector.extend(ee_pos)  # Add XYZ position
            state_vector.extend(ee_orn)  # Add quaternion orientation
            
            return np.array(state_vector, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Failed to get proprioceptive state: {e}")
            # Return dummy state
            return np.zeros(20, dtype=np.float32)

class ActionExtractor:
    """
    Extract actions from robot control
    """
    
    def __init__(self, robot_id: int, controllable_joints: List[int]):
        self.robot_id = robot_id
        self.controllable_joints = controllable_joints
        self.previous_joint_positions = None
    
    def get_current_action(self) -> np.ndarray:
        """
        Get current action (joint position targets or deltas)
        
        Returns:
            np.ndarray: Action vector
        """
        try:
            current_positions = []
            
            # Get current joint targets
            for joint_idx in self.controllable_joints:
                pos, _, _, _ = p.getJointState(self.robot_id, joint_idx)
                current_positions.append(pos)
            
            action = np.array(current_positions, dtype=np.float32)
            
            return action
            
        except Exception as e:
            print(f"❌ Failed to get current action: {e}")
            return np.zeros(7, dtype=np.float32)

# Utility functions for HDF5 data loading and inspection

def inspect_hdf5_file(filename: str):
    """
    Inspect the contents of an HDF5 demonstration file
    
    Args:
        filename: Path to HDF5 file
    """
    try:
        with h5py.File(filename, 'r') as f:
            print(f"📁 Inspecting {filename}")
            print(f"🔍 Top-level groups: {list(f.keys())}")
            
            if 'data' in f:
                data_group = f['data']
                demos = list(data_group.keys())
                print(f"🎬 Found {len(demos)} demonstrations: {demos}")
                
                for demo_name in demos[:3]:  # Show first 3 demos
                    demo = data_group[demo_name]
                    print(f"\n📊 {demo_name}:")
                    print(f"  Groups: {list(demo.keys())}")
                    
                    if 'obs' in demo:
                        obs = demo['obs']
                        print(f"  Observations: {list(obs.keys())}")
                        for obs_name in obs.keys():
                            dataset = obs[obs_name]
                            print(f"    {obs_name}: {dataset.shape} {dataset.dtype}")
                    
                    if 'actions' in demo:
                        actions = demo['actions']
                        print(f"  Actions: {actions.shape} {actions.dtype}")
                    
                    # Show attributes
                    if demo.attrs:
                        print(f"  Metadata: {dict(demo.attrs)}")
                        
    except Exception as e:
        print(f"❌ Failed to inspect {filename}: {e}")

def load_demonstration(filename: str, demo_name: str) -> Dict[str, np.ndarray]:
    """
    Load a specific demonstration from HDF5 file
    
    Args:
        filename: Path to HDF5 file
        demo_name: Name of demonstration to load
        
    Returns:
        Dict containing loaded data arrays
    """
    try:
        with h5py.File(filename, 'r') as f:
            demo_group = f['data'][demo_name]
            
            data = {}
            
            # Load observations
            if 'obs' in demo_group:
                obs_group = demo_group['obs']
                for obs_name in obs_group.keys():
                    data[f'obs_{obs_name}'] = obs_group[obs_name][:]
            
            # Load actions
            if 'actions' in demo_group:
                data['actions'] = demo_group['actions'][:]
            
            # Load trajectory
            if 'traj' in demo_group and 'future' in demo_group['traj']:
                data['traj_future'] = demo_group['traj']['future'][:]
            
            print(f"✅ Loaded {demo_name}: {len(data)} data streams")
            return data
            
    except Exception as e:
        print(f"❌ Failed to load {demo_name} from {filename}: {e}")
        return {}

# Integration functions to work with existing code

def create_integrated_recorder(robot_id: int, 
                              controllable_joints: List[int],
                              camera = None,
                              filename: str = None):
    """
    Create integrated recording system
    
    Returns:
        Tuple of (recorder, state_extractor, action_extractor)
    """
    if camera is None:
        camera = EnhancedWorkspaceCamera()
    
    recorder = HDF5DemonstrationRecorder(filename=filename)
    state_extractor = RobotStateExtractor(robot_id, controllable_joints)
    action_extractor = ActionExtractor(robot_id, controllable_joints)
    
    return recorder, state_extractor, action_extractor

# Backward compatibility imports from utilities2
try:
    from utilities2 import *
except ImportError:
    print("⚠️ Could not import from utilities2, some functions may not be available")