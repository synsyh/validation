import pymongo
from bson.objectid import ObjectId
import numpy as np

from data_trans import trans2matrix, analysis_data, scale, get_velocity


class MongoData:
    def __init__(self, object_id='5bdd2357b87546075830aad1'):
        self.client = pymongo.MongoClient('mongodb://root:Koumin917@192.168.0.2:27017/')
        self.db = self.client['vaptcha']
        self.col = self.db['userdrawpath']
        self.object_id = object_id

    def get_mongodb_batch(self, size=1, if_points=0):
        results = self.col.find({'Similarity': {'$gt': 80}, '_id': {'$gt': ObjectId(self.object_id)}},
                                {'_id': 1, 'VerifyPath': 1, 'Points': 1}).limit(size)
        vps = []
        object_id = ''
        if if_points:
            for result in results:
                vps.append({'VerifyPath': result['VerifyPath'], 'Points': result['Points']})
                # mongodb无法切片，目前是循环获取，保存最后一个id
                object_id = result['_id']
        else:
            for result in results:
                vps.append(result['VerifyPath'])
                # mongodb无法切片，目前是循环获取，保存最后一个id
                object_id = result['_id']
        self.object_id = object_id
        return vps

    def get_mongodb_batch_info(self, size=1):
        results = self.col.find({'Similarity': {'$gt': 60}, '_id': {'$gt': ObjectId(self.object_id)}},
                                {'_id': 1, 'VerifyPath': 1, 'Points': 1, 'DrawTimeString': 1}).limit(size)
        vps = []
        object_id = ''
        for result in results:
            vps.append(result)
            # mongodb无法切片，目前是循环获取，保存最后一个id
            object_id = result['_id']
        self.object_id = object_id
        return vps

    def get_batch_matrix(self, size=32):
        labels = np.zeros((size, 4))

        vps = self.get_mongodb_batch(size, if_points=1)
        datas = np.zeros((len(vps), 128, 128, 3))
        for i, vp in enumerate(vps):
            ps = analysis_data(vp['VerifyPath'])
            ps = scale(ps)
            ps = get_velocity(ps)
            data = trans2matrix(ps)
            datas[i] = data
        return datas


if __name__ == '__main__':
    mongo_data = MongoData()
    datas = mongo_data.get_mongodb_batch(size=1)
    print(datas)
