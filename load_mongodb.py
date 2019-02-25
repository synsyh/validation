import pymongo
from bson.objectid import ObjectId
import numpy as np

from data_trans import trans2matrix, analysis_data, scale, get_velocity


class MongoData:
    def __init__(self, object_id='5b7faf12cc494a07782b1502'):
        self.client = pymongo.MongoClient('mongodb://root:Koumin917@localhost:27017/')
        self.db = self.client['vaptcha']
        self.col = self.db['userdrawpath']
        self.object_id = object_id

    def get_mongodb_batch(self, size=10):
        results = self.col.find({'Similarity': {'$gt': 80}, '_id': {'$gt': ObjectId(self.object_id)}},
                                {'_id': 1, 'VerifyPath': 1, }).limit(size)
        vps = []
        object_id = ''
        for result in results:
            vps.append(result['VerifyPath'])
            # mongodb无法切片，目前是循环获取，保存最后一个id
            object_id = result['_id']
        vps.remove(vps[0])
        self.object_id = object_id
        return vps

    # mongo输入的size有问题，要比输入的少一个
    def get_batch_matrix(self, size=33):
        vps = self.get_mongodb_batch(size)
        datas = np.zeros((len(vps), 128, 128, 3))
        for i, vp in enumerate(vps):
            ps = analysis_data(vp)
            ps = scale(ps)
            ps = get_velocity(ps)
            data = trans2matrix(ps)
            datas[i] = data
        return datas


if __name__ == '__main__':
    mongo_data = MongoData()
    datas = mongo_data.get_batch_matrix()
    print()
