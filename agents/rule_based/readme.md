# Rule-Based Agent

## Folder Structure Overview
```
agents/rule_based/
│
├── videos # Directory to save test run videos
│ └── rule_based-episode-0.mp4 # A test run of Rule-Based agent
├── agent.py  # Rule-Based agent code
├── init_test_rule_based.py # Script to run and test rule-based agent
└──  videos/ # Video output directory
```

## Basics
1. Rule-Based Agent takes in unnormalized observation data.
2. The system framework is similar to the perception and decision layers in a typical L2+ autopilot system, corresponding to two blocks within the code: Safety Estimation and Decision Making. 
3. The Decision Making block can be further divided into two sets of decision logics for actively overtaking and safety. Please check project report and code for details. 

## Decision Logics
1. For every car observed, update frontal afety distances, corner gaps, and lane availability.
2. Decide if left / right lane is free and probable to change lane.
3. Change lane when current x distance is less than minimum or other left right lanes are clearer.
4. Otherwise cruise at target speed through Bang-Bang control.

## Additional Details
1. Implemented lane change cooldown logic to avoid constant lane changing behaviours.
2. Front safe distance changing with agent speed, and immediately increases and gradually decreases in a few steps after lane changing. (IMPORTANT safety estimates)
3. Dynamic and static lane change conditions. 
4. Added **`_act_hy_safe_rule()`** to perform minimum safety check for potential hybrid (RL+rule-based safety check) autonomous driving setup. 

## Key Parameters
1. **`TARGET_SPEED`**: cruise target speed. Default is 30m/s. 
2. **`LANE_CHANGE_COOLDOWN`**: lane change cooldown. Minimum steps between lane changes to mitigate oscillating between lanes. Default is 3.
3. **`TARGET_REACT_TIME`**: target react time. This is used to update frontal safe distance and corner gaps. Default is 1.2s. 

## Others
1. A simple test script that saves an example demo video is at **`RL-AD/agents/rule_based/init_test_rule_based.py`**
2. The evaluation script is at **`RL-AD/test_evaluation_framework.py`**, including comparisons with the DQN agent.
3. 2025/12/06 Update: evaluation scripts for comparisons with SAC and PPO agent are available at **`RL-AD/agents/sac`** and **`RL-AD/agents/ppo`**. 
4. More demo videos are saved in the **`RL-AD/demo/`** folder for visual inspection of agent behavior.