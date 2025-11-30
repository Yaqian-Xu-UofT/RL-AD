# Video save directory: /initial_demo/highway-v0/rule_based-episode-0.mp4
Decision logics (conservative) : 
1. For every car observed, update distances
2. Decide if left / right lane is free and probable to change lane
3. Change lane ONLY WHEN current x distance is less than minimum
4. Otherwise accelerate or decelerate

Additional details: 
1. Lane change cooldown to avoid constant lane changing behaviours
2. Front safe distance changing with agent speed, and decreases in a few steps immediately after lane changing. (important)
3. Dynamic and static lane change conditions. 