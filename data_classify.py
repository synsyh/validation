from load_mongodb import MongoData

mongo_data = MongoData()

n = 5
x = 430
y = 250
grid_x = x / n
grid_y = y / n
grids = {}
count_max = -1


def get_grid_num(p):
    num_x = int(p[0] / grid_x)
    num_y = int(p[1] / grid_y)

    return num_y * n + num_x


for i in range(10):
    vps = mongo_data.get_mongodb_batch(1000, if_points=1)
    for vp in vps:
        points = vp['Points']
        a = [points[0]['X'], points[0]['Y']]
        b = [points[17]['X'], points[17]['Y']]
        c = [points[34]['X'], points[34]['Y']]
        d = [points[50]['X'], points[50]['Y']]

        if get_grid_num(a) < 10:
            tmp = '0' + str(get_grid_num(a))
            code = tmp + str(get_grid_num(b) + 25) + str(get_grid_num(c) + 50) + str(get_grid_num(d) + 75)
        else:
            code = str(get_grid_num(a)) + str(get_grid_num(b) + 25) + str(get_grid_num(c) + 50) + str(
                get_grid_num(d) + 75)
        if code in grids.keys():
            grids[code][len(grids[code])] = vp['VerifyPath']
            grids[code]['length'] += 1
        else:
            grids[code] = {'length': 1}
            grids[code][1] = vp['VerifyPath']

del_list = []
for grid in grids.items():
    if grid[1]['length'] < 10:
        del_list.append(grid[0])
for del_item in del_list:
    del grids[del_item]

for i in range(400):
    print(i)
    vps = mongo_data.get_mongodb_batch(1000, if_points=1)
    for vp in vps:
        points = vp['Points']
        a = [points[0]['X'], points[0]['Y']]
        b = [points[17]['X'], points[17]['Y']]
        c = [points[34]['X'], points[34]['Y']]
        d = [points[50]['X'], points[50]['Y']]

        if get_grid_num(a) < 10:
            tmp = '0' + str(get_grid_num(a))
            code = tmp + str(get_grid_num(b) + 25) + str(get_grid_num(c) + 50) + str(get_grid_num(d) + 75)
        else:
            code = str(get_grid_num(a)) + str(get_grid_num(b) + 25) + str(get_grid_num(c) + 50) + str(
                get_grid_num(d) + 75)
        if code in grids.keys():
            grids[code][len(grids[code])] = vp['VerifyPath']
            grids[code]['length'] += 1
        else:
            continue

with open('./classify_data.txt', 'a') as f:
    for grid in grids.items():
        f.write(str(grid) + '\n')
