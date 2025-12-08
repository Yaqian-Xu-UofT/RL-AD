# Rule-Based Agent

## Folder Structure Overview
agents/rule_based/
│
├── agent.py  # Rule-Based agent code
├── init_test_rule_based.py # Script to run and test rule-based agent
├── videos/ # Video output directory

## Basics
1. Rule-Based Agent takes in unnormalized observation data.
2. The system framework is similar to the perception and decision layers in a typical L2+ autopilot system, corresponding to two blocks within the code: Safety Estimation and Decision Making. 
3. The Decision Making block can be further divided into two sets of decision logics for actively overtaking and safety. Please check project report and code for details. 

## Decision Logics
1. For every car observed, update safety distances, gaps, and lane availability.
2. Decide if left / right lane is free and probable to change lane.
3. Change lane when current x distance is less than minimum or other left right lanes are clearer.
4. Otherwise cruise at target speed.

## Additional Details
1. Lane change cooldown to avoid constant lane changing behaviours.
2. Front safe distance changing with agent speed, and increases in a few steps immediately after lane changing. (important)
3. Dynamic and static lane change conditions. 
4. Added _act_hy_safe_rule() to perform minimum safety check for potential hybrid (RL+rule-based safety check) autonomous driving setup. 

## Key Parameters
1. target_speed: cruise target speed. Default is 30m/s.
2. lccd: lane change cooldown. Minimum steps between lane changes to mitigate oscillating between lanes. Default is 5.
3. trt: target react time. Default is 1.2s. 

## Others
1. Test scripts available at RL-AD/test_evaluation_framework.py, including comparisons with DQN and PPO agents.
2. Scripts for comparisons with SAC agent are available at RL-AD/agents/sac . 