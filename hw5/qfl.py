import random, time

def compute_state_key(pos, play):
    yps = pos[0] / pos[3]
    yfd = pos[2] / pos[1]
    row = 2 if yps > 4 else 1 if yps > 2 else 0
    col = 2 if yfd > 5 else 1 if yfd > 2 else 0
    return (play, row, col)

def best_action(pos, model, q):
    size = model.offensive_playbook_size()
    return max(range(size), key=lambda p: q[compute_state_key(pos, p)])

def q_learn(model, time_limit):
    q_table = {(p, r, c): 0 for p in range(model.offensive_playbook_size()) for r in range(3) for c in range(3)}
    eps, alpha, gamma = 0.2, 0.1, 0.95
    deadline = time.time() + time_limit
    size = model.offensive_playbook_size()
    while time.time() < deadline:
        pos = model.initial_position()
        while not model.game_over(pos):
            if random.random() < eps: action = random.randrange(size)
            else: action = best_action(pos, model, q_table)
            nxt = model.result(pos, action)[0]
            over = model.game_over(nxt)
            reward = 0 if not over else (1 if model.win(nxt) else -1)
            key = compute_state_key(pos, action)
            if over: q_table[key] += alpha * (reward - q_table[key])
            else:
                next_key = compute_state_key(nxt, best_action(nxt, model, q_table))
                q_table[key] += alpha * (reward + gamma * q_table[next_key] - q_table[key])
            pos = nxt
            eps *= 0.999999
            alpha *= 0.999999
    return lambda position: best_action(position, model, q_table)

