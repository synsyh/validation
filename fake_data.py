from load_mongodb import MongoData
from data_trans import analysis_data, scale, get_velocity
from data_vector import trans2vector
import random

mongo_data = MongoData()
vps = mongo_data.get_mongodb_batch(size=1)
for vp in vps:
    raw_data = analysis_data(vp)
    points = scale(raw_data)
    fake_points = points
    for i in range(len(fake_points) - 1):
        i += 1
        if fake_points[i - 1]['x'] != fake_points[i + 1]['x']:
            x_drift = random.randint(fake_points[i - 1]['x'], fake_points[i + 1]['x'])
        else:
            x_drift = 0
        if fake_points[i - 1]['y'] != fake_points[i + 1]['y']:
            y_drift = random.randint(fake_points[i - 1]['y'], fake_points[i + 1]['y'])
        else:
            y_drift = 0
        time_drift = random.randint(0, 10)
        fake_points[i]['x'] = x_drift
        fake_points[i]['y'] = y_drift
        fake_points[i]['time'] += time_drift
    fake_points = get_velocity(fake_points)
    points = get_velocity(points)
    vector_ps = trans2vector(points)
    vector_fps = trans2vector(fake_points)
    print(vector_ps)
