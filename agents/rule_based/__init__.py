import numpy as np

class RuleBasedAgent:
    def __init__(self, env, target_speed=30.0):
        self.env = env
        self.target_speed = target_speed
        
    def act(self, observation):
        # Check if config exists, otherwise assume discrete
        if hasattr(self.env.unwrapped, "config") and "action" in self.env.unwrapped.config:
            action_type = self.env.unwrapped.config["action"]["type"]
            if action_type == "DiscreteMetaAction":
                return self._act_discrete(observation)
            elif action_type == "ContinuousAction":
                return self._act_continuous(observation)
        
        # Default to discrete if unknown
        return self._act_discrete(observation)
    
    def crash_detection(self, observation):
        # Placeholder for crash detection logic
        pass

    def _act_discrete(self, observation):
        # Actions
        LANE_LEFT = 0
        IDLE = 1
        LANE_RIGHT = 2
        FASTER = 3
        SLOWER = 4
        
        # Parse observation
        # Assuming Kinematics observation: [V, F]
        # Row 0 is ego vehicle
        
        if len(observation.shape) != 2:
            # Fallback for unexpected observation shape
            return IDLE

        # Ego vehicle features
        # 0: presence, 1: x, 2: y, 3: vx, 4: vy
        # In default highway-env Kinematics:
        # x: longitudinal position
        # y: lateral position
        # vx: longitudinal velocity
        # vy: lateral velocity
        
        # Find front vehicle in same lane
        front_dist = 100
        left_lane_free = True
        right_lane_free = True
        FR_DIST_TH = 30  # Threshold distance to front
        
        for i in range(1, len(observation)):
            if observation[i, 0] == 0: # Not present
                print("Skipping non-present vehicle")
                continue
            
            dx = observation[i, 1] # relative x (if absolute=False)
            dy = observation[i, 2] # relative y
            
            # Check if same lane (approximate)
            if abs(dy) < 3:
                if dx > 0 and dx < front_dist:
                    front_dist = dx
            
            # Check adjacent lanes
            if abs(dx) < FR_DIST_TH:
                if dy < 0 and dy > -10:
                    right_lane_free = False
                elif dy > 0 and dy < 10:
                    left_lane_free = False
        print(f"Front dist: {front_dist}, Left free: {left_lane_free}, Right free: {right_lane_free}, Ego speed: {observation[0,3]}")
        

        # Decision logic
        action = IDLE
        
        text_action = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

        # Speed control: 
        # Lane change if blocked
        if  front_dist <= FR_DIST_TH:
            if left_lane_free and right_lane_free:
                action = LANE_LEFT if np.random.rand() < 0.5 else LANE_RIGHT
            elif left_lane_free:
                action = LANE_LEFT
            elif right_lane_free:
                action = LANE_RIGHT
            else:
                action = SLOWER
        else:
            action = FASTER
        print(f"Action: {text_action[action]}")
        return action

    def _act_continuous(self, observation):
        # Placeholder for continuous action space
        # Throttle a and steering angle delta
        return np.array([0.0, 0.0])
