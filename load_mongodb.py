import pymongo
from bson.objectid import ObjectId


class MongoData:
    def __init__(self, object_id='5b7faf12cc494a07782b1502'):
        self.client = pymongo.MongoClient('mongodb://root:Koumin917@192.168.0.2:27017/')
        self.db = self.client['vaptcha']
        self.col = self.db['userdrawpath']
        self.object_id = object_id

    def get_mongodb_batch(self, size=10):
        results = self.col.find({'Similarity': {'$gt': 80}, '_id': {'$gt': ObjectId(self.object_id)}},
                                {'_id': 1, 'VerifyPath': 1, }).limit(size)
        verify_paths = []
        object_id = ''
        for result in results:
            verify_paths.append(result['VerifyPath'])
            # mongodb无法切片，目前是循环获取，保存最后一个id
            object_id = result['_id']
        verify_paths.remove(verify_paths[0])
        self.object_id = object_id

        return verify_paths


if __name__ == '__main__':
    mongo_data = MongoData()
    for i in range(3):
        verify_paths = mongo_data.get_mongodb_batch()
        for path in verify_paths:
            print(path)
        print()
