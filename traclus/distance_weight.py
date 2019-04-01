import json

with open('./data/test_output.json') as f:
    reps = json.loads(f.read())
# with open('./data/json_data_30k_3000.json') as f:
#     clus_datas = json.loads(f.read())['trajectories']

with open('./data/test_output_clusters.json') as f:
    cls = json.loads(f.read())

tmp = [len(i) for i in cls]
print(min(tmp))
