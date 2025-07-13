# sequential-planning-of-robotic-arm-through-imitation-learning
Project Summary
This project focuses on enabling sequential planning and control of a simulated robotic arm through Imitation Learning (IL). Using Franka Emika Panda 7-DOF robotic arm simulated in PyBullet, we demonstrate how manual teleoperation data can be collected and used to train a deep neural network model that learns to reproduce the sequential joint movements required for task execution.

The robot learns from demonstration data (trajectories of joint positions and visual observations), collected through manual keyboard teleoperation, and stores them in HDF5 files. These demonstrations are then used to train a CNN + LSTM model that predicts the robot’s next joint configurations.

