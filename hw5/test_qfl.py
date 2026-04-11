import time
import sys
import random
import nfl_strategy as nfl

import qfl

# defenses are 4-3 Blast Man-to-man, 4-3 Sam-Will Blitz, Prevent
# outcomes are (yards, time, turnover)

_slant_37 = [[(-1, 4, False), (20, 3, False), (10, 2, False), (7, 2, False), (4, 2, False)],
           [(1, 3, False), (10, 2, False), (5, 2, False), (1, 4, False), (2, 2, False)],
           [(8, 2, False), (20, 3, False), (4, 2, False), (2, 2, False), (6, 2, False)]]

_y_cross_80 = [[(0, 1, True), (-7, 4, False), (0, 1, False), (0, 1, True), (24, 2, False)],
              [(0, 1, False), (0, 1, True), (-6, 4, False), (18, 2, False), (13, 2, False)],
              [(19, 4, False), (0, 1, True), (22, 2, False), (47, 3, False), (0, 1, False)]]

_z_corner_82 = [[(28, 3, False), (-9, 4, False), (47, 3, False), (0, 1, False), (0, 1, False)],
               [(30, 3, False), (0, 1, True), (8, 2, False), (-8, 4, False), (0, 1, False)],
               [(38, 3, False), (-9, 4, False), (0, 1, False), (0, 1, True), (0, 1, False)]]

game_parameters = [
    ([_slant_37, _y_cross_80, _z_corner_82], [0.20, 0.05, 0.25, 0.1, 0.4]),
    ([_y_cross_80, _z_corner_82, _slant_37], [0.05, 0.20, 0.1, 0.25, 0.4])
    ]


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("USAGE: {python3 | pypy3}", sys.argv[0], "model-number learning-time num-games")
        sys.exit(1)

    try:
        game = int(sys.argv[1])
        limit = float(sys.argv[2])
        n = int(sys.argv[3])
    except:
        print("USAGE: {python3 | pypy3}", sys.argv[0], "model-number learning-time num-games")
        sys.exit(1)

    model = nfl.NFLStrategy(*game_parameters[game])
    start = time.time();
    policy = qfl.q_learn(model, limit);
    t = time.time() - start
    if t > limit * 1.001:
        print("WARNING: Q-learning ran for", t, "seconds; allowed", limit)
    
    print(model.simulate(policy, n))
