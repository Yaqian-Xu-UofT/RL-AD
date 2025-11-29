import numpy as np
from highway_env.vehicle.kinematics import Vehicle

class RuleBasedAgent:
    def __init__(self, env, target_speed=30.0):
        self.env = env
        self.TARGET_SPEED = target_speed
        self.LEFT_LANE_Y = -2
        self.RIGHT_LANE_Y = (self.env.unwrapped.config["lane_count"]-1) * 4  # Assuming lane width of 4m
        self.VEHICLE_LENGTH = Vehicle.LENGTH
        self.LANE_CHANGE_COOLDOWN = 5  # Minimum steps between lane changes
        self.cooldown_counter = self.LANE_CHANGE_COOLDOWN  # Initialize to allow immediate lane change
        self.lane_change_cooled = self.cooldown_counter >= self.LANE_CHANGE_COOLDOWN
        print(self.RIGHT_LANE_Y)
        
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

    def _act_discrete(self, observation):
        # Actions
        LANE_LEFT = 0
        IDLE = 1
        LANE_RIGHT = 2
        FASTER = 3
        SLOWER = 4
        
        # Parse observation

        # Ego vehicle features
        # 0: presence, 1: x, 2: y, 3: vx, 4: vy
        # In default highway-env Kinematics:
        # x: longitudinal position
        # y: lateral position
        # vx: longitudinal velocity
        # vy: lateral velocity
        ego_x = observation[0, 1]
        ego_y = observation[0, 2]
        ego_vx = observation[0, 3]
        ego_vy = observation[0, 4]
        ego_spd = np.sqrt(ego_vx**2 + ego_vy**2)

        # Safe distance to front vehicle
        FRONT_SAFE_DIST = 30 + np.power(ego_spd, 1.6) * 0.15
        FRONT_SAFE_DIST -= (self.LANE_CHANGE_COOLDOWN - self.cooldown_counter)*self.VEHICLE_LENGTH*1.25
        FRONT_SAFE_DIST = max(FRONT_SAFE_DIST, 30)
        LANE_CHANGE_FRONT_SAFE = 35 + np.power(ego_spd, 1.6) * 0.15
        LANE_CHANGE_REAR_SAFE = 5
        LANE_WIDTH = 4

        # Cooldown logic
        if self.cooldown_counter >= self.LANE_CHANGE_COOLDOWN:
            self.lane_change_cooled = True
        else:
            self.lane_change_cooled = False
            self.cooldown_counter += 1

        # Check road boundaries
        can_go_left = ego_y > self.LEFT_LANE_Y + LANE_WIDTH*0.8
        can_go_right = ego_y < self.RIGHT_LANE_Y - LANE_WIDTH*0.8

        # 1. Analyze surrounding vehicles
        # Initialize
        dist_cur_lane = 200
        dist_left_lane = 200
        dist_right_lane = 200
        left_lane_free = True
        right_lane_free = True

        
        for i in range(1, len(observation)):
            if observation[i, 0] == 0: # Not present
                continue
            dx = observation[i, 1] - ego_x # relative x
            dy = observation[i, 2] - ego_y # relative y
            dvx = observation[i, 3]
            dvy = observation[i, 4]
            
            # Check vehicle in the same lane within front safe distance
            if abs(dy) < LANE_WIDTH / 2:
                if dx > -self.VEHICLE_LENGTH:
                    dist_cur_lane = min(dist_cur_lane, dx)
            
            # Check left lane
            if -1.5 * LANE_WIDTH < dy < -0.2 * LANE_WIDTH:
                if -LANE_CHANGE_REAR_SAFE < dx < LANE_CHANGE_FRONT_SAFE:
                    if dx > -self.VEHICLE_LENGTH:
                        dist_left_lane = min(dx, dist_left_lane)
                    left_lane_free = False
                    
            # Check right lane
            if 0.2 * LANE_WIDTH < dy < 1.5 * LANE_WIDTH:
                if -LANE_CHANGE_REAR_SAFE < dx < LANE_CHANGE_FRONT_SAFE:
                    if dx > -self.VEHICLE_LENGTH:
                        dist_right_lane = min(dx, dist_right_lane)
                    right_lane_free = False

        print(f"Front dist: {dist_cur_lane:<7.2f} "
              f"FNT safe: {FRONT_SAFE_DIST:<7.2f} "
              f"LFT free: {str(left_lane_free):<6} "
              f"RGT free: {str(right_lane_free):<6} "
              f"AG spd: {observation[0,3]:<6.2f} "
              f"AG y: {ego_y:<6.2f} "
              f"Can go left: {str(can_go_left):<6} "
              f"Can go right: {str(can_go_right):<6}"
              f"CD: {self.cooldown_counter:<2} "
              f"CD?: {self.lane_change_cooled}")
        

        # Decision logic
        action = IDLE

        # If there is a vehicle too close at front, consider lane change
        if dist_cur_lane < FRONT_SAFE_DIST:
            options = []
            if can_go_left and left_lane_free and self.lane_change_cooled:
                options.append(LANE_LEFT)
            if can_go_right and right_lane_free and self.lane_change_cooled:
                options.append(LANE_RIGHT)
            if len(options) == 0:
                action = SLOWER
            elif len(options) == 1:
                action = options[0]
            else:
                if dist_left_lane > dist_right_lane:
                    action = LANE_LEFT
                elif dist_right_lane > dist_left_lane:
                    action = LANE_RIGHT
                else:
                    action = np.random.choice(options)

        else:
            options = []
            if ego_vx < self.TARGET_SPEED - 2:
                action = FASTER
            elif ego_vx > self.TARGET_SPEED + 2:
                action = SLOWER
            else:
                if can_go_left and left_lane_free and dist_left_lane > FRONT_SAFE_DIST + 10:
                    options.append(LANE_LEFT)
                elif can_go_right and right_lane_free and dist_right_lane > FRONT_SAFE_DIST + 10:
                    options.append(LANE_RIGHT)
                if len(options) == 0:
                    action = IDLE
                elif len(options) == 1:
                    action = options[0]
                else:
                    action = np.random.choice(options)
        
        # reset cooldown if lane change is taken
        if action in [LANE_LEFT, LANE_RIGHT]:
            self.lane_change_cooled = False
            self.cooldown_counter = 0

        text_action = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
        print(f"Action: {text_action[action]}")
        return action

    def _act_continuous(self, observation):
        # Placeholder for continuous action space
        # Throttle a and steering angle delta
        return np.array([0.0, 0.0])
