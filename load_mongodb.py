import pymongo
from bson.objectid import ObjectId

myclient = pymongo.MongoClient("mongodb://root:Koumin917@192.168.0.2:27017/")
mydb = myclient["vaptcha"]
mycol = mydb["userdrawpath"]


def get_mongodb_batch(object_id, size=10):
    results = mycol.find({'Similarity': {'$gt': 80}, '_id': {'$gt': ObjectId(object_id)}},
                         {'_id': 1, 'VerifyPath': 1, }).limit(size)
    verify_paths = []
    for result in results:
        verify_paths.append(result['VerifyPath'])
        # mongodb无法切片，目前是循环获取，保存最后一个
        object_id = result['_id']
    verify_paths.remove(verify_paths[0])
    return verify_paths, object_id


if __name__ == '__main__':
    verify_paths, object_id = get_mongodb_batch('5b7faf12cc494a07782b1502')
    for path in verify_paths:
        print(path)
