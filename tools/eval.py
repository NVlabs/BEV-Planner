import numpy as np
import mmcv
# data = mmcv.load('/mount/data/jiahan/fbbev/test/Sun_Oct_15_11_35/results_nusc_planning.json')
data = mmcv.load('/mount/data/FBBEV/test/planner_r50_8x4_12ep_102x102_4f_S111_fix2_/Tue_Oct_24_03_58/results_nusc_planning.json')
#sort
keys = list(data.keys())
# print(keys)
new_keys = []
for key in keys:
   s =key.split("-")
   new_keys.append([int(s[1]),int(s[2])])

new_keys=sorted(new_keys,key=(lambda x:(x[0], x[1])))
sorted_keys = []
for key in new_keys:
   v = ['scene',  str(key[0]).zfill(4), str(key[1]) ]
   k='-'.join(v)
   sorted_keys.append(k)

print(len(data))

all_scene_keys=[]
key='-'.join(sorted_keys[0].split("-")[:2])
# print(key)
scene=[]

for k in sorted_keys:
    if(key in k):
        # print(True)
        scene.append(k)
    else:
        s =k.split("-")
        key='-'.join(s[:2])
        if len(scene)<39:
            print(scene)
        all_scene_keys.append(scene)
        scene=[k]

# print(all_scene_keys)
len(all_scene_keys)
#tranform raw data
new_data={}
for keys in all_scene_keys:
    l = len(keys)
    for i in range(l):
        val = []
        index = i
        for j in range(i+1):
            if index>6:
                index-=1
            else:
                val.append(data[keys[j]][index])
                index-=1
        new_data[keys[i]]=val

#compute mean and var
stable_dist_with_gt=[]
stable_mean_distance=[]
stable_variance_distance=[]

for key, value in new_data.items():
    #filter unstable data
    if(len(value)!=7):
        continue
    assert len(value)==7
    #compute mean
    gt = value[-1]
    pred = value[:-1]
    coor_mean= np.mean(pred, axis=0)
    #L2
    dist = np.linalg.norm(coor_mean - gt)
    stable_dist_with_gt.append(dist)

    #compute var
    data_array = np.array(pred)
    
    distances = np.linalg.norm(data_array - coor_mean, axis=1)
    mean_distance = np.mean(distances)
    variance_distance = np.var(distances)

    stable_mean_distance.append(mean_distance)
    stable_variance_distance.append(variance_distance)

print('stable_dist_with_gt: {}'.format(np.mean(stable_dist_with_gt)))
print('stable_mean_distance: {}'.format(np.mean(stable_mean_distance)))
print('stable_variance_distance: {}'.format(np.mean(stable_variance_distance)))

import random
import math
import matplotlib.pyplot as plt

# 生成40种不同颜色的列表
colors = ['#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255) ) for _ in range(40)]

# colors = ['g', 'b', 'c', 'm', 'y', 'k', 'purple', 'orange', 'pink', 'brown', 'lime', 'teal', 'gold', 'indigo', 'slategray', 'violet', 'darkred', 'maroon', 'orchid']  # 20种不同颜色选项
markers = ['x', 's', 'D', '^', 'v', 'p']  # 不同的标记选项

for keys in all_scene_keys:
    all_coor=[]
    l = len(keys)

    for i in range(l):
        coordinates=data[keys[i]]
        all_coor.extend(coordinates)
    min_x=min(coor[0] for coor in all_coor)
    min_y=min(coor[1] for coor in all_coor)
    max_x=max(coor[0] for coor in all_coor)
    max_y=max(coor[1] for coor in all_coor)  
    ratio=math.ceil((max_y-min_y)/(max_x-min_x))

    plt.figure(figsize=(8, 8*ratio), dpi=300)
    plt.gca().invert_yaxis()  # 反转y轴，将原点移至左上角

    gt_traj=[]
    for i in range(l):
        coordinates=data[keys[i]]
        x_coords, y_coords = zip(*coordinates)
        gt_traj.append(coordinates[0])
        color = colors[i % len(colors)]
        plt.scatter(x_coords[0], y_coords[0], s=15, marker='o',c='r')
        # plt.scatter(x_coords[1:], y_coords[1:], s=15, marker='o',c=color)
        for j in range(len(coordinates) - 1):
            if i+j > l-2: 
                break
            plt.plot([x_coords[j], x_coords[j + 1]], [y_coords[j], y_coords[j + 1]], '-',c=color, linewidth=0.5)  
    
    x_gt_coords, y_gt_coords = zip(*gt_traj)
    for i in range(len(gt_traj) - 1):
        plt.plot([x_gt_coords[i], x_gt_coords[i + 1]], [y_gt_coords[i], y_gt_coords[i + 1]], '-',c='r', linewidth=1)  
    plt.axis('equal') 
    
    for i in range(l):

        col_coordinates=new_data[keys[i]]
        x_coords, y_coords = zip(*col_coordinates)
        color = colors[i % len(colors)]
        for j in range(len(col_coordinates)-1):
            marker = markers[j % len(markers) ]
            # plt.plot([x_coords[j], x_coords[j + 1]], [y_coords[j], y_coords[j + 1]], '-',c=color, linewidth=0.5)  
            plt.scatter(x_coords[j], y_coords[j], s=10, marker=marker,c=color)


    plt.xlabel('X')
    plt.ylabel('Y')
    s =keys[0].split("-")
    key='-'.join(s[:2])
    plt.savefig(f'../{key}_111_fix2.png')
    print(key)
    plt.close()