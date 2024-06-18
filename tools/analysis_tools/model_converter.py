import torch

model = torch.load('/mount/data/FBBEV/work_dirs/mappetrv3_noaug_8x8_36ep_102x102/iter_31644_ema.pth')


keys = list(model['state_dict'].keys())

for k in keys:
    model['state_dict'][k.replace('pts_bbox_head', 'uni_perceive_head')] = model['state_dict'][k]

torch.save(model, '/mount/data/FBBEV/work_dirs/mappetrv3_noaug_8x8_36ep_102x102/iter_31644_ema2.pth')