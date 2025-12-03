import numpy as np
from highway_env.vehicle.kinematics import Vehicle

class RuleBasedAgent:
    def __init__(self, env, target_speed=30.0):
        self.env = env
        self.TARGET_SPEED = target_speed

        self.LANE_WIDTH = 4
        self.LEFT_LANE_Y = -2
        self.RIGHT_LANE_Y = (self.env.unwrapped.config["lane_count"]-1) * self.LANE_WIDTH
        self.VEHICLE_LENGTH = Vehicle.LENGTH
        self.LANE_CHANGE_COOLDOWN = 3  # Minimum steps between lane changes. Default 5
        self.cooldown_counter = 0  # Initialize to allow immediate lane change
        self.lane_change_cooled = True
        print(self.RIGHT_LANE_Y)
        
    def act(self, observation, episode=None):
        # Check if config exists, otherwise assume discrete
        if hasattr(self.env.unwrapped, "config") and "action" in self.env.unwrapped.config:
            action_type = self.env.unwrapped.config["action"]["type"]
            if action_type == "DiscreteMetaAction":
                return self._act_discrete(observation, episode=episode)
            # elif action_type == "ContinuousAction":
            #     return self._act_continuous(observation, episode=episode)
        
        # Default to discrete if unknown
        return self._act_discrete(observation, episode=episode)

    def _act_discrete(self, observation, episode=None):
        # Actions
        LANE_LEFT = 0
        IDLE = 1
        LANE_RIGHT = 2
        FASTER = 3
        SLOWER = 4

        # Ego vehicle features
        ego_x = observation[0, 1]
        ego_y = observation[0, 2]
        ego_vx = observation[0, 3]
        ego_vy = observation[0, 4]
        ego_spd = np.sqrt(ego_vx**2 + ego_vy**2)

        # Driver related parameters
        TARGET_REACT_TIME = 1.2
        MIN_STATIC_GAP = self.VEHICLE_LENGTH*1.001
        FRONT_STATIC_GAP = MIN_STATIC_GAP
        REAR_STATIC_GAP = MIN_STATIC_GAP + 0.25*self.VEHICLE_LENGTH


        ###################################
        ##       SAFETY ESTIMATION       ##
        ###################################

        FRONT_SAFE_DIST = MIN_STATIC_GAP + ego_spd * TARGET_REACT_TIME
        dist_factor = (self.LANE_CHANGE_COOLDOWN - self.cooldown_counter)
        FRONT_SAFE_DIST -= (dist_factor / self.LANE_CHANGE_COOLDOWN)*(FRONT_SAFE_DIST - REAR_STATIC_GAP)
        # # FRONT_SAFE_DIST -= (self.LANE_CHANGE_COOLDOWN - self.cooldown_counter)*self.VEHICLE_LENGTH*0.3 # *1.25
        FRONT_SAFE_DIST = max(FRONT_SAFE_DIST, FRONT_STATIC_GAP)

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        self.lane_change_cooled = (self.cooldown_counter == 0)

        can_go_left = ego_y > self.LEFT_LANE_Y + self.LANE_WIDTH*0.8
        can_go_right = ego_y < self.RIGHT_LANE_Y - self.LANE_WIDTH*0.8

        dist_cur_lane = 200
        dist_left_lane = 200
        dist_right_lane = 200
        left_lane_free = True
        right_lane_free = True

        vel_cur_lane = 100
        vel_left_lane = 100
        vel_right_lane = 100

        safety_gap = np.zeros(4) # 0: LF, 1: RF, 2: LR, 3: RR
        for i in range(1, len(observation)):
            if observation[i, 0] == 0: # Not present
                continue
            dx = observation[i, 1] - ego_x # relative x
            dy = observation[i, 2] - ego_y # relative y
            vx = observation[i, 3]
            vy = observation[i, 4]
            dvx = observation[i, 3] - ego_vx
            dvy = observation[i, 4] - ego_vy

            # Vehicle in current lane
            if abs(dy) < self.LANE_WIDTH / 2:
                if dx > -self.VEHICLE_LENGTH:
                    if dx > 0:
                        dist_cur_lane = min(dist_cur_lane, dx)
                        if dx < 60:
                            vel_cur_lane = min(vel_cur_lane, vx)
            # Vehicle in left lane
            if -1.5 * self.LANE_WIDTH < dy < -0.5 * self.LANE_WIDTH:
                if dx > -self.VEHICLE_LENGTH:
                    if dx > 0:
                        dist_left_lane = min(dist_left_lane, dx)
                        if dx < 60:
                            vel_left_lane = min(vel_left_lane, vx)
                # Left cars in front
                if dx > 0:
                    safety_gap[0] = FRONT_STATIC_GAP - TARGET_REACT_TIME * dvx
                    safety_gap[0] = max(safety_gap[0], MIN_STATIC_GAP)
                    if dx < safety_gap[0]:
                        left_lane_free = False
                # Left cars in rear
                else:
                    if -dx < REAR_STATIC_GAP:
                        right_lane_free = False
                    safety_gap[2] = REAR_STATIC_GAP + TARGET_REACT_TIME * dvx
                    safety_gap[2] = max(safety_gap[2], MIN_STATIC_GAP)
                    if -dx < safety_gap[2]:
                        left_lane_free = False
            # Vehicle in right lane
            if 0.5 * self.LANE_WIDTH < dy < 1.5 * self.LANE_WIDTH:
                if dx > -self.VEHICLE_LENGTH:
                    if dx > 0:
                        dist_right_lane = min(dist_right_lane, dx)
                        if dx < 60:
                            vel_right_lane = min(vel_right_lane, vx)
                # Right cars in front
                if dx > 0:
                    safety_gap[1] = FRONT_STATIC_GAP - TARGET_REACT_TIME * dvx
                    safety_gap[1] = max(safety_gap[1], MIN_STATIC_GAP)
                    if dx < safety_gap[1]:
                        right_lane_free = False
                # Right cars in rear
                else:
                    if -dx < REAR_STATIC_GAP:
                        right_lane_free = False
                    safety_gap[3] = REAR_STATIC_GAP + TARGET_REACT_TIME * dvx
                    safety_gap[3] = max(safety_gap[3], MIN_STATIC_GAP)
                    if -dx < safety_gap[3]:
                        right_lane_free = False


        ###################################
        ##     DECISION MAKING LOGIC     ##
        ###################################

        action = IDLE
        front_blocked = (dist_cur_lane < FRONT_SAFE_DIST) or (vel_cur_lane < self.TARGET_SPEED-5 and dist_cur_lane < 40)

        if front_blocked:
            options = []
            if can_go_left and left_lane_free and self.lane_change_cooled:
                options.append(LANE_LEFT)
            if can_go_right and right_lane_free and self.lane_change_cooled:
                options.append(LANE_RIGHT)
            if len(options) == 0:
                if dist_cur_lane < FRONT_SAFE_DIST:
                    action = SLOWER
                else:
                    action = IDLE
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
            if ego_vx < self.TARGET_SPEED - 0.5:
                action = FASTER
            elif ego_vx > self.TARGET_SPEED:
                action = SLOWER
            if self.TARGET_SPEED - 0.5 <= ego_vx <= self.TARGET_SPEED:
                options = []
                if self.lane_change_cooled and can_go_left and left_lane_free and dist_left_lane > dist_cur_lane + 10:
                    options.append(LANE_LEFT)
                if self.lane_change_cooled and can_go_right and right_lane_free and dist_right_lane > dist_cur_lane + 10:
                    options.append(LANE_RIGHT)
                if len(options) == 0:
                    action = IDLE
                elif len(options) == 1:
                    action = options[0]
                else:
                    action = np.random.choice(options)

        if action in [LANE_LEFT, LANE_RIGHT]:
            self.lane_change_cooled = False
            self.cooldown_counter = self.LANE_CHANGE_COOLDOWN

        disp_safety_gap = "[" + ", ".join([f"{x:.2f}" for x in safety_gap]) + "]"

        print(f"Episode: {episode} "
              f"Front dist: {dist_cur_lane:<7.2f} "
              f"FNT safe: {FRONT_SAFE_DIST:<7.2f} "
              f"LC gap: {disp_safety_gap:<25} "
              f"LFT free: {str(left_lane_free):<6} "
              f"RGT free: {str(right_lane_free):<6} "
              f"AG spd: {ego_vx:<6.2f} "
              f"AG y: {ego_y:<5.2f} "
              f"Can go left: {str(can_go_left):<6} "
              f"Can go right: {str(can_go_right):<6}"
              f"CD: {self.cooldown_counter:<2} "
              f"CD?: {self.lane_change_cooled}")
        
        text_action = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
        print(f"Action: {text_action[action]}")
        return action







    def _act_continuous(self, observation, episode=None):
        # Placeholder for continuous action space
        # Throttle a and steering angle delta
        return np.array([0.0, 0.0])
