import math

import pymongo
from bson import ObjectId

from data_trans import analysis_data, scale, get_velocity
from load_mongodb import MongoData


class DataCleaner:
    def __init__(self):
        self.mongo_data = MongoData('5b7fae6ccc494a07782b1485')
        self.client = pymongo.MongoClient('mongodb://root:Koumin917@192.168.0.2:27017/')
        self.db = self.client['vaptcha']
        self.t_col = self.db['error_backup_time']
        self.p_col = self.db['error_backup_position']
        self.a_col = self.db['error_backup_acceleration']

    def time_error(self, points):
        for j in range(len(points) - 1):
            if points[j]['time'] == points[j + 1]['time']:
                tmp = self.mongo_data.col.find({'_id': ObjectId(self.mongo_data.object_id)})
                for i in tmp:
                    print(i)
                    self.t_col.insert_one(i)
                self.mongo_data.col.remove(self.mongo_data.object_id)
                break

    def position_error(self, points):
        points = scale(points)
        for j in range(len(points) - 1):
            dis_x = points[j + 1]['x'] - points[j]['x']
            dis_y = points[j + 1]['y'] - points[j]['y']
            dis = math.sqrt(dis_x ** 2 + dis_y ** 2)
            if dis > 80:
                tmp = self.mongo_data.col.find({'_id': ObjectId(self.mongo_data.object_id)})
                for i in tmp:
                    print(i)
                    self.p_col.insert_one(i)
                self.mongo_data.col.remove(self.mongo_data.object_id)
                break

    # def acceleration_error(self, points):
    #     points = scale(points)
    #     points = get_velocity(points)
    #     for j in range(len(points) - 1):
    #         a = (points[j + 1]['v'] - points[j]['v']) * 1000 / (points[j + 1]['time'] - points[j]['time'])
    #         if a > 20:
    #             break

    def clean(self):
        for i in range(100000):
            verify_paths = self.mongo_data.get_mongodb_batch()
            for path in verify_paths:
                points = analysis_data(path)
                # self.time_error(points)
                self.position_error(points)
        print(self.mongo_data.object_id)


dc = DataCleaner()
dc.clean()
