from bson import ObjectId
from pymongo import MongoClient

client = MongoClient('mongodb://root:Koumin917@192.168.0.2:27017/')
db = client['vaptcha']
col = db['userdrawpath']
col1 = db['backup']

# query = {'VaptchaCellId': '5b7cda24fc650e163c72aeb2'}
# col.delete_many(query)
# col.remove(ObjectId('5b7fae6ecc494a07782b1487'))
a = col1.find({'_id': ObjectId('5b7fae6ccc494a07782b1485')})
for i in a:
    print(i)
