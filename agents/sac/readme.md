# job-id: 108132 (mildly overtaking, fast speed, less collision)
1. train_sb3_multicore.py
2. test_sb3_108132.py
3. Env Config
   {'action': {'type': 'ContinuousAction', 'longitudinal': True, 'lateral': True, 'steering_range': [np.float64(-0.2617993877991494), np.float64(0.2617993877991494)], 'dynamical': True, 'clip': True}, 'lanes_count': 3, 'duration': 60, 'observation': {'type': 'Kinematics', 'vehicles_count': 15, 'features': ['presence', 'x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'], 'normalize': True, 'absolute': False}, 'lane_change_reward': 1, 'collision_reward': -0.1, 'right_lane_reward': 0.0, 'high_speed_reward': 2.5, 'reward_speed_range': [30, 35], 'vehicle_density': 1.5, 'offroad_terminal': True, 'normalize_reward': False}
4. train_steps: 1.6M

# job-id: 108430 (aggressively overtaking, fast speed, more collision)
1. train_sb3_multicore_overtake.py
2. test_sb3_108430.py
3. Env Config
   {'action': {'type': 'ContinuousAction', 'longitudinal': True, 'lateral': True, 'steering_range': [np.float64(-0.2617993877991494), np.float64(0.2617993877991494)], 'dynamical': True, 'clip': True}, 'lanes_count': 4, 'duration': 60, 'observation': {'type': 'Kinematics', 'vehicles_count': 15, 'features': ['presence', 'x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'], 'normalize': True, 'absolute': False}, 'lane_change_reward': 1, 'collision_reward': 0.0, 'right_lane_reward': 0.0, 'high_speed_reward': 2.5, 'reward_speed_range': [30, 35], 'vehicle_density': 1.5, 'offroad_terminal': True, 'normalize_reward': False}
4. train_steps: 1.6M

# job-id: 108437 (same as job-id 108132 + CustomIDMVehicle)
1. train_sb3_multicore_custom.py
2. test_sb3_108437.py
3. Env Config
   {'action': {'type': 'ContinuousAction', 'longitudinal': True, 'lateral': True, 'steering_range': [np.float64(-0.2617993877991494), np.float64(0.2617993877991494)], 'dynamical': True, 'clip': True}, 'lanes_count': 3, 'duration': 60, 'observation': {'type': 'Kinematics', 'vehicles_count': 15, 'features': ['presence', 'x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'], 'normalize': True, 'absolute': False}, 'lane_change_reward': 1, 'collision_reward': -0.1, 'right_lane_reward': 0.0, 'high_speed_reward': 2.5, 'reward_speed_range': [30, 35], 'vehicle_density': 1.5, 'offroad_terminal': True, 'normalize_reward': False, 'other_vehicles_type': 'custom.CustomIDMVehicle'}
5. train_steps: 1M

# saved model in ./ckpt
# saved video in ./video_sb3
