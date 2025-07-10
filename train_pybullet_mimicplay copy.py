import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import cv2
from utilities2 import *  # Original utilities
from utilities3 import *  # HDF5 recording utilities
from enhanced_camera import EnhancedWorkspaceCamera

class RobotCNNLSTM(nn.Module):
    """
    CNN+LSTM architecture for robot control from visual and proprioceptive input
    
    Architecture:
    1. CNN processes RGB images to extract visual features
    2. Visual features + proprioceptive data feed into LSTM
    3. LSTM output processed by MLP to predict joint positions
    """
    
    def __init__(self, 
                 image_channels=3,
                 proprio_dim=20,
                 action_dim=7,
                 lstm_hidden_dim=256,
                 cnn_feature_dim=512,
                 mlp_hidden_dim=128):
        super(RobotCNNLSTM, self).__init__()
        
        self.image_channels = image_channels
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.cnn_feature_dim = cnn_feature_dim
        
        # CNN for image feature extraction
        self.cnn = nn.Sequential(
            # Input: [batch, seq, channels, height, width]
            # Need to reshape for CNN processing
            
            # Layer 1: 3 -> 32 channels
            nn.Conv2d(image_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            
            # Layer 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 3: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Additional layers for larger images
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Adaptive pooling to ensure fixed output size
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        
        # Calculate CNN output dimensions (empirically)
        self._cnn_output_dim = 128 * 7 * 7  # Based on adaptive pooling to 7x7
        
        # Linear layer to reduce CNN features
        self.cnn_to_features = nn.Sequential(
            nn.Flatten(),  # Flatten CNN output
            nn.Linear(self._cnn_output_dim, cnn_feature_dim),
            nn.ReLU()
        )
        
        # LSTM for temporal dynamics
        # Input: CNN features + proprioceptive data
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim + proprio_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,  # 2-layer LSTM for more capacity
            batch_first=True,  # Input: [batch, seq, features]
            dropout=0.2
        )
        
        # MLP head for action prediction
        self.action_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, action_dim)
        )
        
    def forward(self, rgb, proprio):
        """
        Forward pass through the network
        
        Args:
            rgb: [batch, seq, channel, height, width] RGB images
            proprio: [batch, seq, proprio_dim] Proprioceptive features
            
        Returns:
            actions: [batch, seq, action_dim] Predicted actions
        """
        batch_size, seq_len = rgb.shape[0], rgb.shape[1]
        
        # Process images with CNN
        # Reshape to process all images at once
        rgb_reshaped = rgb.reshape(batch_size * seq_len, self.image_channels, rgb.shape[3], rgb.shape[4])
        
        # Pass through CNN and feature extractor
        cnn_features = self.cnn(rgb_reshaped)
        visual_features = self.cnn_to_features(cnn_features)
        
        # Reshape back to [batch, seq, feature]
        visual_features = visual_features.reshape(batch_size, seq_len, self.cnn_feature_dim)
        
        # Concatenate with proprioceptive data
        lstm_input = torch.cat([visual_features, proprio], dim=2)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        # Predict actions (process all timesteps)
        actions = self.action_head(lstm_out)
        
        return actions
    
    def predict_single_step(self, rgb, proprio, hidden_state=None):
        """
        Make a single step prediction (for inference)
        
        Args:
            rgb: [batch, channel, height, width] RGB image
            proprio: [batch, proprio_dim] Proprioceptive features
            hidden_state: Previous LSTM hidden state (h, c) or None
            
        Returns:
            action: [batch, action_dim] Predicted action
            new_hidden_state: Updated LSTM hidden state
        """
        batch_size = rgb.shape[0]
        
        # Add sequence dimension for consistency
        if len(rgb.shape) == 4:
            rgb = rgb.unsqueeze(1)  # [batch, 1, channel, height, width]
        if len(proprio.shape) == 2:
            proprio = proprio.unsqueeze(1)  # [batch, 1, proprio_dim]
        
        # Get visual features
        cnn_features = self.cnn(rgb.reshape(batch_size, self.image_channels, rgb.shape[3], rgb.shape[4]))
        visual_features = self.cnn_to_features(cnn_features).unsqueeze(1)  # [batch, 1, cnn_feature_dim]
        
        # Combine with proprio
        lstm_input = torch.cat([visual_features, proprio], dim=2)
        
        # LSTM step
        if hidden_state is None:
            lstm_out, new_hidden_state = self.lstm(lstm_input)
        else:
            lstm_out, new_hidden_state = self.lstm(lstm_input, hidden_state)
        
        # Predict action
        action = self.action_head(lstm_out.squeeze(1))
        
        return action, new_hidden_state


class AIRobotController:
    """AI-powered robot controller using trained CNN+LSTM model"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.hidden_state = None
        self.prediction_history = []
        self.confidence_threshold = 0.1
        self.max_joint_change = 0.05  # Maximum joint change per step for safety
        
        # Load normalization parameters
        self.action_mean = None
        self.action_std = None
        self.proprio_mean = None
        self.proprio_std = None
        
        # Control parameters
        self.control_active = False
        self.last_prediction_time = 0
        self.prediction_interval = 0.1  # Predict every 100ms
        
        print(f"🤖 AI Robot Controller initialized on {self.device}")
        
    def load_model(self):
        """Load the trained model and normalization parameters"""
        try:
            self._load_normalization_params()
            actual_proprio_dim = 20  # Default
            if self.proprio_mean is not None:
               actual_proprio_dim = len(self.proprio_mean)
               print(f"🔍 Detected proprioceptive dimension: {actual_proprio_dim}")

            # Load model
            self.model = RobotCNNLSTM(
                image_channels=3,
                proprio_dim=20,
                action_dim=7,
                lstm_hidden_dim=256,
                cnn_feature_dim=512,
                mlp_hidden_dim=128
            )
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully from {self.model_path}")
            
            # Load normalization parameters
            self._load_normalization_params()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def _load_normalization_params(self):
        """Load normalization parameters from HDF5 file"""
        try:
            # Try to find a normalized HDF5 file
            normalized_file = "E:\\roboarmsimulation\\robot_demos_20250708_115502\demonstrations_11_normalized.hdf5"
            
            if os.path.exists(normalized_file):
                with h5py.File(normalized_file, 'r') as f:
                    if 'normalization' in f:
                        if 'action_mean' in f['normalization']:
                            self.action_mean = f['normalization']['action_mean'][:]
                            self.action_std = f['normalization']['action_std'][:]
                        if 'proprio_mean' in f['normalization']:
                            self.proprio_mean = f['normalization']['proprio_mean'][:]
                            self.proprio_std = f['normalization']['proprio_std'][:]
                        
                        print("✅ Normalization parameters loaded")
                        return True
            
            # Fallback: use default normalization
            print("⚠️ Using default normalization parameters")
            self.action_mean = np.zeros(7)
            self.action_std = np.ones(7)
            self.proprio_mean = np.zeros(20)
            self.proprio_std = np.ones(20)
            
        except Exception as e:
            print(f"❌ Error loading normalization params: {e}")
            # Use safe defaults
            self.action_mean = np.zeros(7)
            self.action_std = np.ones(7)
            self.proprio_mean = np.zeros(20)
            self.proprio_std = np.ones(20)
    
    def preprocess_image(self, rgb_image):
        """Preprocess RGB image for model input"""
        try:
            # Convert to numpy if needed
            if isinstance(rgb_image, tuple):
                rgb_image = rgb_image[0]
            
            # Ensure correct shape and type
            if rgb_image.shape[-1] == 4:  # RGBA
                rgb_image = rgb_image[:, :, :3]  # Remove alpha channel
            
            # Normalize to [0, 1]
            rgb_image = rgb_image.astype(np.float32) / 255.0
            
            # Convert to CHW format for PyTorch
            rgb_image = np.transpose(rgb_image, (2, 0, 1))
            
            # Convert to tensor and add batch dimension
            rgb_tensor = torch.tensor(rgb_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            return rgb_tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ Image preprocessing error: {e}")
            return None
    
    def preprocess_proprioceptive(self, proprio_data):
        """Preprocess proprioceptive data for model input"""
        try:
            # Ensure numpy array
            if isinstance(proprio_data, list):
                proprio_data = np.array(proprio_data, dtype=np.float32)
            
            # Debug: Print dimensions to understand the mismatch
            print(f"🔍 Proprio data shape: {proprio_data.shape}, Model expects: 20")
            
            # Handle dimension mismatch
            if len(proprio_data) > 20:
                # Trim to first 20 dimensions
                proprio_data = proprio_data[:20]
                print(f"⚠️ Trimmed proprioceptive data from {len(proprio_data)} to 20 dimensions")
            elif len(proprio_data) < 20:
                # Pad with zeros
                padding = np.zeros(20 - len(proprio_data))
                proprio_data = np.concatenate([proprio_data, padding])
                print(f"⚠️ Padded proprioceptive data from {len(proprio_data)} to 20 dimensions")
            
            # Normalize if parameters available
            if self.proprio_mean is not None and self.proprio_std is not None:
                # Ensure normalization parameters match the data dimensions
                if len(self.proprio_mean) == len(proprio_data):
                    proprio_data = (proprio_data - self.proprio_mean) / self.proprio_std
                else:
                    print(f"⚠️ Normalization parameter mismatch - using raw data")
            
            # Convert to tensor and add batch and sequence dimensions
            proprio_tensor = torch.tensor(proprio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            return proprio_tensor.to(self.device)
        
        except Exception as e:
            print(f"❌ Proprioceptive preprocessing error: {e}")
            return None
    
    def predict_action(self, rgb_image, proprio_data):
        """Predict robot action from current observation"""
        if self.model is None:
            return None
        
        try:
            # Preprocess inputs
            rgb_tensor = self.preprocess_image(rgb_image)
            proprio_tensor = self.preprocess_proprioceptive(proprio_data)
            
            if rgb_tensor is None or proprio_tensor is None:
                return None
            
            # Make prediction
            with torch.no_grad():
                if self.hidden_state is None:
                    # First prediction
                    action_pred = self.model(rgb_tensor, proprio_tensor)
                    action = action_pred.squeeze().cpu().numpy()
                else:
                    # Use previous hidden state for continuity
                    action, self.hidden_state = self.model.predict_single_step(
                        rgb_tensor.squeeze(1), proprio_tensor.squeeze(1), self.hidden_state
                    )
                    action = action.squeeze().cpu().numpy()
            
            # Denormalize action
            if self.action_mean is not None and self.action_std is not None:
                action = action * self.action_std + self.action_mean
            
            # Safety check: limit joint changes
            action = self._apply_safety_limits(action)
            
            # Store prediction for analysis
            self.prediction_history.append(action.copy())
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)
            
            return action
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def _apply_safety_limits(self, action):
        """Apply safety limits to prevent dangerous movements"""
        try:
            # Clip joint angles to safe ranges (Panda robot limits)
            joint_limits = [
                (-2.8973, 2.8973),  # Joint 0
                (-1.7628, 1.7628),  # Joint 1
                (-2.8973, 2.8973),  # Joint 2
                (-3.0718, -0.0698), # Joint 3
                (-2.8973, 2.8973),  # Joint 4
                (-0.0175, 3.7525),  # Joint 5
                (-2.8973, 2.8973),  # Joint 6
            ]
            
            # Apply joint limits
            for i, (min_val, max_val) in enumerate(joint_limits):
                if i < len(action):
                    action[i] = np.clip(action[i], min_val, max_val)
            
            return action
            
        except Exception as e:
            print(f"❌ Safety limit error: {e}")
            return action
    
    def execute_action(self, robot_id, controllable_joints, action):
        """Execute predicted action on robot"""
        try:
            if action is None or len(action) == 0:
                return False
            
            # Apply action to joints
            for i, joint_pos in enumerate(action):
                if i < len(controllable_joints):
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=controllable_joints[i],
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_pos,
                        force=500,
                        maxVelocity=1.0  # Controlled movement
                    )
            
            return True
            
        except Exception as e:
            print(f"❌ Action execution error: {e}")
            return False
    
    def get_confidence_score(self):
        """Calculate confidence score based on prediction consistency"""
        if len(self.prediction_history) < 2:
            return 0.0
        
        # Calculate variance in recent predictions
        recent_predictions = np.array(self.prediction_history[-5:])
        variance = np.mean(np.var(recent_predictions, axis=0))
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = 1.0 / (1.0 + variance)
        return confidence
    
    def reset_state(self):
        """Reset internal state"""
        self.hidden_state = None
        self.prediction_history = []
        print("🔄 AI controller state reset")


def main():
    """Main AI robot control loop"""
    
    # Connect to PyBullet
    try:
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print("✅ Connected to PyBullet successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to PyBullet: {e}")
        exit(1)
    
    # Setup simulation (same as simulationtest53.py)
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
    
    # Create systems
    gripper = AutoGraspingSystem(robot_id, controllable_joints, box_ids, color_names)
    state_extractor = RobotStateExtractor(robot_id, controllable_joints)
    
    # Initialize camera
    workspace_camera = EnhancedWorkspaceCamera(
        position=[1.0, -0.8, 0.7],
        target=[0.0, 0.0, 0.1],
        width=320,
        height=240,
        debug=False
    )
    
    # Initialize AI controller
    model_path = "E:\\roboarmsimulation\imitationlearning\\robot_cnn_lstm_best4o.pth"  # Update with your model path
    ai_controller = AIRobotController(model_path, device='cuda')
    
    # Load model
    if not ai_controller.load_model():
        print("❌ Failed to load AI model - exiting")
        p.disconnect()
        exit(1)
    
    # Control variables
    ai_control_active = False
    manual_control_active = False
    last_prediction_time = 0
    prediction_interval = 0.1  # Predict every 100ms
    
    # Manual control setup (same as original)
    key_to_joint = {
        ord('q'): (0, +1), ord('a'): (0, -1),  # Joint 1
        ord('w'): (1, +1), ord('s'): (1, -1),  # Joint 2
        ord('e'): (2, +1), ord('d'): (2, -1),  # Joint 3
        ord('r'): (3, +1), ord('f'): (3, -1),  # Joint 4
        ord('t'): (4, +1), ord('g'): (4, -1),  # Joint 5
        ord('y'): (5, +1), ord('h'): (5, -1),  # Joint 6
        ord('u'): (6, +1), ord('j'): (6, -1),  # Joint 7
    }
    
    print("\n🤖 AI ROBOT CONTROL SYSTEM")
    print("   1 = Manual Control | 2 = AI Control | 3 = Both")
    print("   7 = Home | 8 = Ready | 9 = Short Floor | 0 = Reset")
    print("   Z = Toggle Auto-Grasp | X = Release Gripper")
    print("   C = Show AI Confidence | V = Reset AI State")
    print("   Manual: Q/A=J0, W/S=J1, E/D=J2, R/F=J3, T/G=J4, Y/H=J5, U/J=J6")
    print("   ESC = Exit")
    print("\n🧠 AI Model loaded and ready!")
    
    frame_counter = 0
    
    # Main control loop
    while True:
        frame_counter += 1
        current_time = time.time()
        
        # Process keyboard events
        keys = p.getKeyboardEvents()
        
        for key, state in keys.items():
            if state & p.KEY_WAS_TRIGGERED:
                if key == 27:  # ESC
                    print("🛑 Exit requested")
                    p.disconnect()
                    exit(0)
                
                elif key == ord('1'):
                    ai_control_active = False
                    manual_control_active = True
                    ai_controller.reset_state()
                    print("🕹️ Manual control mode")
                
                elif key == ord('2'):
                    ai_control_active = True
                    manual_control_active = False
                    ai_controller.reset_state()
                    print("🧠 AI control mode")
                
                elif key == ord('3'):
                    ai_control_active = True
                    manual_control_active = True
                    ai_controller.reset_state()
                    print("🤖 Hybrid control mode (AI + Manual)")
                
                elif key == ord('7'):
                    print("🏠 Moving to HOME...")
                    move_to_preset(robot_id, controllable_joints, 'home')
                    ai_controller.reset_state()
                
                elif key == ord('8'):
                    print("🚀 Moving to READY...")
                    move_to_preset(robot_id, controllable_joints, 'ready')
                    ai_controller.reset_state()
                
                elif key == ord('9'):
                    print("📦 Moving to SHORT FLOOR...")
                    move_to_preset(robot_id, controllable_joints, 'short_floor')
                    ai_controller.reset_state()
                
                elif key == ord('0'):
                    print("🔄 Resetting robot position...")
                    initialize_robot_joints(robot_id, controllable_joints)
                    ai_controller.reset_state()
                
                elif key == ord('z'):
                    try:
                        if hasattr(gripper, 'enabled'):
                            gripper.enabled = not gripper.enabled
                            status = "ON" if gripper.enabled else "OFF"
                            print(f"🤏 Auto-grasping: {status}")
                    except Exception as e:
                        print(f"❌ Auto-grasping toggle error: {e}")
                
                elif key == ord('x'):
                    try:
                        success = gripper.manual_release()
                        if success:
                            print("✅ Gripper released")
                        else:
                            print("❌ Gripper release failed")
                    except Exception as e:
                        print(f"❌ Gripper release error: {e}")
                
                elif key == ord('c'):
                    confidence = ai_controller.get_confidence_score()
                    print(f"🎯 AI Confidence: {confidence:.3f}")
                
                elif key == ord('v'):
                    ai_controller.reset_state()
                    print("🔄 AI state reset")
        
        # Manual control
        if manual_control_active:
            for key, state in keys.items():
                if state & p.KEY_IS_DOWN:  # Changed from KEY_WAS_TRIGGERED to KEY_IS_DOWN for continuous control
                    if key in key_to_joint:
                        joint_idx, direction = key_to_joint[key]
                        if joint_idx < len(controllable_joints):
                            joint_id = controllable_joints[joint_idx]
                            current_pos, _, _, _ = p.getJointState(robot_id, joint_id)
                            new_pos = current_pos + direction * 0.03
                            
                            p.setJointMotorControl2(
                                bodyUniqueId=robot_id,
                                jointIndex=joint_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=new_pos,
                                force=500,
                                maxVelocity=2.0
                            )
        
        # AI control
        if ai_control_active and (current_time - last_prediction_time) > prediction_interval:
            try:
                # Capture current observation
                rgb_image = workspace_camera.capture_image()
                proprio_data = state_extractor.get_proprioceptive_state()
                
                if rgb_image is not None and proprio_data is not None:
                    # Predict action
                    predicted_action = ai_controller.predict_action(rgb_image, proprio_data)
                    
                    if predicted_action is not None:
                        # Execute action
                        success = ai_controller.execute_action(robot_id, controllable_joints, predicted_action)
                        
                        if success:
                            confidence = ai_controller.get_confidence_score()
                            
                            # Show prediction info occasionally
                            if frame_counter % 50 == 0:
                                print(f"🧠 AI Prediction: Confidence={confidence:.3f}")
                                print(f"    Joints: {[f'{x:.3f}' for x in predicted_action[:3]]}...")
                        
                        last_prediction_time = current_time
                
            except Exception as e:
                print(f"❌ AI control error: {e}")
        
        # Update systems
        gripper.update_auto_grasping()
        
        # Status display
        if frame_counter % 300 == 0:
            ai_status = "🧠" if ai_control_active else "⚫"
            # manual_status = "🕹️" if manual_control_active else "⚫"
            confidence = ai_controller.get_confidence_score()
            
            print(f"📊 AI:{ai_status}  | Confidence:{confidence:.3f}")
        
        # Step simulation
        p.stepSimulation()
        time.sleep(1./240.)


if __name__ == "__main__":
    main()