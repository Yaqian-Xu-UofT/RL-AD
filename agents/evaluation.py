def evaluation(num_collision, num_completed, num_overtakes, total_episodes):
    # Evaluate performance metrics
    # what else?
    collision_rate = num_collision / total_episodes
    completion_rate = num_completed / total_episodes
    overtake_rate = num_overtakes / total_episodes
    print(f"Collision Rate: {collision_rate:.2f}, Completion Rate: {completion_rate:.2f}, Overtake Rate: {overtake_rate:.2f}")
    return collision_rate, completion_rate, overtake_rate