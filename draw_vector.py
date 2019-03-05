from data_trans import analysis_data, scale, get_velocity
from data_vector import trans2vector
from load_mongodb import MongoData

mongo_data = MongoData()
vps = mongo_data.get_mongodb_batch(size=1)
vp = vps[0]
points = analysis_data(vp)
points = scale(points, if_zero=True)
points = get_velocity(points)
vector_points = trans2vector(points)
print()
