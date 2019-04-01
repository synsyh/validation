from data_trans import analysis_data
from load_mongodb import MongoData
from traclus.line_segment import Point
from traclus.line_segment_compute import get_trajectory_line_segments_from_points_iterable
import matplotlib.pyplot as plt
import numpy as np

mongo_data = MongoData()
vps = mongo_data.get_mongodb_batch_info(size=1)
for vp in vps:
    plt.figure()
    vp = vp['VerifyPath']
    points = analysis_data(vp)
    traj = [Point(i['x'], i['y']) for i in points]
    tmp = get_trajectory_line_segments_from_points_iterable(traj)
    line_segments = tmp.line_segment
    for ls in line_segments:
        plt.plot((ls.start.x, ls.start.y), (ls.end.x, ls.end.y))
    plt.show()
    plt.figure()
    xs = np.asarray([p['x'] for p in points])
    ys = np.asarray([p['y'] for p in points])
    plt.plot(xs, ys)
    plt.show()
print()
