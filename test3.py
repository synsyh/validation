import json

tmp = {
    "trajectories": [[{'angle_dif': 0.99, 'v': 1.22}, {'angle_dif': 0.99, 'v': 1.22}, {'angle_dif': 0.99, 'v': 1.25}],
                     [{'angle_dif': 0.99, 'v': 1.22}, {'angle_dif': 0.99, 'v': 1.22}, {'angle_dif': 0.99, 'v': 1.25}]]}

with open('tmp.txt', 'w') as f:
    json.dump(tmp, f)

