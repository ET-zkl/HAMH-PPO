import json
import math

with open('anon_3_4_jinan_real.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
with open('anon_3_4_jinan_real.json', 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)
up_inters = ['road_4_4_1', 'road_3_4_1', 'road_2_4_1', 'road_1_4_1']
below_inters = ['road_1_1_3', 'road_2_1_3', 'road_3_1_3', 'road_4_1_3']
right_inters = ['road_4_4_0', 'road_4_3_0', 'road_4_2_0', 'road_4_1_0']
left_inters = ['road_1_1_2', 'road_1_2_2', 'road_1_3_2', 'road_1_4_2']
sn1 = 0
sn2 = 0
sn3 = 0
sn0_7 = 0
we = [0, 0, 0, 0]  # 从西到东
for item in data:
    if item['route'][0] == 'road_1_0_1' and item['route'][-1] == 'road_6_1_0':# or item['route'][0] =='road_4_5_3' and item['route'][-1] in below_inters:
        sn0_7 = sn0_7 + 1
    if item['route'][0] == 'road_1_0_1' and item['route'][-1] == 'road_1_1_1':
        sn3 = sn3+1
    if item['route'][0] == 'road_2_0_1' and item['route'][-1] in up_inters or item['route'][0] =='road_2_5_3' and item['route'][-1] in below_inters:
        sn2 = sn2+1
    if item['route'][0] == 'road_1_0_1' and item['route'][-1] in up_inters or item['route'][0] =='road_1_5_3' and item['route'][-1] in below_inters:
        sn1 = sn1+1
    if item['route'][0] == 'road_0_1_0' and item['route'][-1] in right_inters or item['route'][0] == 'road_5_1_2' and item['route'][-1] in left_inters:
        we[0]=we[0]+1
    if item['route'][0] == 'road_0_2_0' and item['route'][-1] in right_inters or item['route'][0] == 'road_5_2_2' and item['route'][-1] in left_inters:
        we[1]+=1
    if item['route'][0] == 'road_0_3_0' and item['route'][-1] in right_inters or item['route'][0] == 'road_5_3_2' and item['route'][-1] in left_inters:
        we[2]+=1
    if item['route'][0] == 'road_0_4_0' and item['route'][-1] in right_inters or item['route'][0] == 'road_5_4_2' and item['route'][-1] in left_inters:
        we[3]+=1
# print(sn0_7)
# print(we)

print('------------------')
all_route = []
all_route_vehicle = {}
vehicle_route = []

# for i in data:
#     if i['route'] not in all_route:
#         all_route.append(i['route'])
#         # print(str(i['route']))
#         all_route_vehicle.setdefault(str(i['route']), 0)
#     else:
#         all_route_vehicle[str(i['route'])] = all_route_vehicle[str(i['route'])] + 1
# for j in all_route_vehicle.items():
    # print(j)
# up_inters = ['road_4_4_1', 'road_3_4_1', 'road_2_4_1', 'road_1_4_1']
# below_inters = ['road_1_1_3', 'road_2_1_3', 'road_3_1_3', 'road_4_1_3']
# right_inters = ['road_4_4_0', 'road_4_3_0', 'road_4_2_0', 'road_4_1_0']
# left_inters = ['road_1_1_2', 'road_1_2_2', 'road_1_3_2', 'road_1_4_2']
# sn1 = 0
# sn2 = 0
# sn3 = 0
# sn4 = 0
# we = [0, 0, 0, 0]  # 从西到东
# for item in data:
#     if item['route'][0] == 'road_4_0_1' and item['route'][-1] in up_inters or item['route'][0] =='road_4_5_3' and item['route'][-1] in below_inters:
#         sn4 = sn4 + 1
#     if item['route'][0] == 'road_3_0_1' and item['route'][-1] in up_inters or item['route'][0] =='road_3_5_3' and item['route'][-1] in below_inters:
#         sn3 = sn3+1
#     if item['route'][0] == 'road_2_0_1' and item['route'][-1] in up_inters or item['route'][0] =='road_2_5_3' and item['route'][-1] in below_inters:
#         sn2 = sn2+1
#     if item['route'][0] == 'road_1_0_1' and item['route'][-1] in up_inters or item['route'][0] =='road_1_5_3' and item['route'][-1] in below_inters:
#         sn1 = sn1+1
#     if item['route'][0] == 'road_0_1_0' and item['route'][-1] in right_inters or item['route'][0] == 'road_5_1_2' and item['route'][-1] in left_inters:
#         we[0]=we[0]+1
#     if item['route'][0] == 'road_0_2_0' and item['route'][-1] in right_inters or item['route'][0] == 'road_5_2_2' and item['route'][-1] in left_inters:
#         we[1]+=1
#     if item['route'][0] == 'road_0_3_0' and item['route'][-1] in right_inters or item['route'][0] == 'road_5_3_2' and item['route'][-1] in left_inters:
#         we[2]+=1
#     if item['route'][0] == 'road_0_4_0' and item['route'][-1] in right_inters or item['route'][0] == 'road_5_4_2' and item['route'][-1] in left_inters:
#         we[3]+=1
# print(sn1,sn2, sn3, sn4)
# print(we)
# with open('hangzhou_4x4_gudang_18041610_1h.json', 'r', encoding='utf-8') as f:
with open('anon_3_4_jinan_real.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for i in data:
    if i['route'] not in all_route:
        all_route.append(i['route'])
        # print(str(i['route']))
        all_route_vehicle.setdefault(str(i['route']), 1)
    else:
        all_route_vehicle[str(i['route'])] = all_route_vehicle[str(i['route'])] + 1

num = 0
for j in all_route_vehicle.items():
    num += j[1]
    print(j)
print(num)

print('每个时刻有多少辆车')
all_time = []
all_time_vehicle = {}
vehicle_time = []
for t in data:
    if t['startTime'] not in all_time:
        all_time.append(t['startTime'])
        all_time_vehicle.setdefault(t['startTime'], 1)
    else:
        all_time_vehicle[t['startTime']] = all_time_vehicle[t['startTime']] + 1

print('每300s来了多少车，共分成12个300s')
arrival_time = []
n = 12  # 数组的长度
arrival_time = [0] * n
for k, v in all_time_vehicle.items():
    t = math.floor(k/300)  # 向下取整
    arrival_time[t] = arrival_time[t] + v

import numpy as np
print(arrival_time)
print('最大值：',np.max(arrival_time))
print('最小值：',np.min(arrival_time))
print('均值：',np.mean(arrival_time))
print('总体方差：',np.var(arrival_time))
print('标准差：',np.std(arrival_time))

print('每个路口经过多少辆车')
all_road = []
all_road_vehicle = {}
vehicle_road = []
for t in data:
    for i in t['route']:
        road = i[:8]
        if road not in all_road:
            all_road.append((road))
            all_road_vehicle.setdefault(road, 1)
        else:
            all_road_vehicle[road] = all_road_vehicle[road]+1
print()
v = [0]*49
n = 1
for i in range(1, 4):
    for j in range(1, 5):
        s = 'road' + '_'+str(i)+'_'+str(j)
        v[n] = [all_road_vehicle[s]]
        n += 1
print(v)
# import torch
# x = torch.tensor([-1, -3], dtype=float)
# # p = torch.tensor([0.25, 0.75], dtype=float)
# p = torch.softmax(x, dim=0)
# log_pro = torch.log(p)
# e = - torch.sum(p * log_pro)
#
# print(x)
# print(torch.softmax(x, dim=0))