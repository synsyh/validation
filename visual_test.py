import json

import matplotlib.pyplot as plt
import numpy as np

with open('./tmp_data/raw_campus_trajectories.txt') as f:
    parsed_input = json.loads(f.read())

points = parsed_input['trajectories']
plt.figure()
for point in points:
    xs = np.asarray([k['x'] for k in point])
    ys = np.asarray([k['y'] for k in point])
    plt.plot(xs, ys, )
plt.show()
