import numpy as np
from functools import partial
import torch
from lib.timer import Timer
from lib.utils import load_obj, natural_key
from datasets.indoor import IndoorDataset
from datasets.modelnet import get_train_datasets, get_test_datasets
import os,re,sys,json,yaml,random, argparse, torch, pickle
from easydict import EasyDict as edict
from configs.models import architectures
from models.architectures import KPFCNN
import open3d as o3d
import matplotlib.pyplot as plt
# import open3d.visualization.jupyter as o3d_jupyter
import plotly.graph_objects as go

def load_config(path):
    """
    Loads config file:

    Args:
        path (str): path to the config file

    Returns: 
        config (dict): dictionary of the configuration parameters, merge sub_dicts

    """
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)
    
    config = dict()
    for key, value in cfg.items():
        for k,v in value.items():
            config[k] = v

    return config

  # load configs
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/train/indoor.yaml', help= 'Path to the config file.')
args = parser.parse_args()
config = load_config(args.config)
config = edict(config)

# snapshot=os.path.join(f'{config.dir}',f'{config.benchmark}',f'{config.num_gpus}_gpu_{config.img_num}_img_initmode_{config.init_mode}_{config.first_feats_dim}_gamma{config.scheduler_gamma}_lr{config.lr}_finalfeatsdim_{config.final_feats_dim}',f'{config.mode}')
# print(f"save results to {snapshot}")
# config['snapshot_dir'] = f'{snapshot}'
# config['tboard_dir'] = f'{snapshot}/tensorboard'
# config['save_dir'] = f'{snapshot}/checkpoints'
# config = edict(config)

# os.makedirs(config.snapshot_dir, exist_ok=True)
# os.makedirs(config.save_dir, exist_ok=True)
# os.makedirs(config.tboard_dir, exist_ok=True)
# json.dump(
#     config,
#     open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
#     indent=4,
# )
if torch.cuda.is_available():
        config.device = torch.device('cuda')
else:
        config.device = torch.device('cpu')
    
    # backup the files
# os.system(f'cp -r models {config.snapshot_dir}')
# os.system(f'cp -r datasets {config.snapshot_dir}')
# os.system(f'cp -r lib {config.snapshot_dir}')
# os.system(f'cp -r configs {config.snapshot_dir}')
# shutil.copy2('main.py',config.snapshot_dir)
# shutil.copy2('configs/train/indoor.yaml',config.snapshot_dir)

    # model initialization
config.architecture = architectures[config.dataset]
config.model = KPFCNN(config)


    # create optimizer 
# if config.optimizer == 'SGD':
#     config.optimizer = optim.SGD(
#         config.model.parameters(), 
#         lr=config.lr,
#         momentum=config.momentum,
#         weight_decay=config.weight_decay,
#         )
# elif config.optimizer == 'ADAM':
#     config.optimizer = optim.Adam(
#         config.model.parameters(), 
#         lr=config.lr,
#         betas=(0.9, 0.999),
#         weight_decay=config.weight_decay,
#     )
    
    # create learning rate scheduler
# config.scheduler = optim.lr_scheduler.ExponentialLR(
#     config.optimizer,
#     gamma=config.scheduler_gamma,
# )

def get_datasets(config):
    info_train = load_obj(config.train_info)
    train_set = IndoorDataset(info_train,config,data_augmentation=True)
    return train_set

# def save_2d_ply(filename, points_2d):
#     with open(filename, 'w') as f:
#         # 写入头部信息
#         f.write("ply\n")
#         f.write("format ascii 1.0\n")
#         f.write(f"element vertex {len(points_2d)}\n")
#         f.write("property float x\n")
#         f.write("property float y\n")
#         f.write("property float z\n")  # 这里添加一个 z 属性，值全为 0
#         f.write("end_header\n")
        
#         # 写入点云数据
#         for point in points_2d:
#             f.write(f"{point[0]} {point[1]} 0.0\n")

def save_2d_ply(filename, points_2d, colors=None):
    with open(filename, 'w') as f:
        # 写入头部信息
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_2d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")  # 添加一个 z 属性，值全为 0
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点云数据
        for i, point in enumerate(points_2d):
            line = f"{point[0]} {point[1]} 0.0"
            if colors is not None:
                color = colors[i]
                line += f" {int(color[0] * 255)} {int(color[1] * 255)} {int(color[2] * 255)}"
            f.write(line + "\n")

if __name__ == '__main__':
    train_set = get_datasets(config)
    # print(train_set[500])
    point = train_set[0]

        # 创建源点云
    src_pcd = point['src_pcd']
    # 创建目标点云
    tgt_pcd = point['tgt_pcd']

    # 如果有颜色数据
    src_colors = point['src_color1']

    tgt_colors = point['tgt_color1']

    src_colors = src_colors.numpy().reshape(-1, 3)
    tgt_colors = tgt_colors.numpy().reshape(-1, 3)

    # 创建 Open3D 点云对象
    src_pcd_o3d = o3d.geometry.PointCloud()
    src_pcd_o3d.points = o3d.utility.Vector3dVector(src_pcd)
    src_pcd_o3d.colors = o3d.utility.Vector3dVector(src_colors)

    tgt_pcd_o3d = o3d.geometry.PointCloud()
    tgt_pcd_o3d.points = o3d.utility.Vector3dVector(tgt_pcd)
    tgt_pcd_o3d.colors = o3d.utility.Vector3dVector(tgt_colors)

    # 可视化点云
    # o3d.visualization.draw_geometries([src_pcd_o3d, tgt_pcd_o3d])
    o3d.io.write_point_cloud("./ply/point_cloud_src_pcd_o3d.ply", src_pcd_o3d)
    o3d.io.write_point_cloud("./ply/point_cloud_tgt_pcd_o3d.ply", tgt_pcd_o3d)

#=================================================================================
    #     # 从数据中提取三维点云数据
    # src_pcd = np.array(point['src_pcd'])
    # tgt_pcd = np.array(point['tgt_pcd'])

    # # 选择投影到 XY 平面（忽略 Z 轴）
    # src_pcd_2d = src_pcd[:, :2]
    # tgt_pcd_2d = tgt_pcd[:, :2]

    # # 保存为 PLY 文件
    # save_2d_ply("./ply/point_cloud_src_pcd_2d.ply", src_pcd_2d)
    # save_2d_ply("./ply/point_cloud_tgt_pcd_2d.ply", tgt_pcd_2d)
#=================================================================================
    # # 从数据中提取三维点云数据
    # src_pcd = np.array(point['src_pcd'])
    # tgt_pcd = np.array(point['tgt_pcd'])

    # # 提取二维点云数据
    # src1_inds2d = np.array(point['src1_inds2d'])
    # tgt1_inds2d = np.array(point['tgt1_inds2d'])

    # # 根据二维索引提取点云数据
    # src_pcd_2d = src_pcd[src1_inds2d[:, 0].astype(int), :2]
    # tgt_pcd_2d = tgt_pcd[tgt1_inds2d[:, 0].astype(int), :2]

    # # 提取颜色数据
    # src_colors = np.array(point['src_color1']).reshape(-1, 3)
    # tgt_colors = np.array(point['tgt_color1']).reshape(-1, 3)

    # src_pcd_2d = src1_inds2d.reshape(-1, 2)
    # tgt_pcd_2d = tgt1_inds2d.reshape(-1, 2)

    # # 保存为 PLY 文件
    # save_2d_ply("./ply/point_cloud_src_pcd_2d.ply", src_pcd_2d,src_colors)
    # save_2d_ply("./ply/point_cloud_tgt_pcd_2d.ply", tgt_pcd_2d,tgt_colors)
#=================================================================================

    # # 假设 src1_inds2d 是你的二维点云数据
    # src1_inds2d = point['src1_inds2d']

    # # 转换为 numpy 数组
    # src1_inds2d_np = src1_inds2d.numpy()

    # # 提取 x 和 y 坐标
    # x = src1_inds2d_np[:, 0]
    # y = src1_inds2d_np[:, 1]

    # # 绘制点云
    # plt.figure(figsize=(6, 6))
    # plt.scatter(x, y, c='blue', s=10)  # 使用蓝色点绘制
    # # plt.grid(True)

    # # 保存图像为 PNG 文件
    # plt.savefig("./ply/src1_inds2d.png", dpi=300)
    # print("2D点云图像已保存为 src1_inds2d.png")

    #=================================================================================
    pcd = o3d.io.read_point_cloud("./ply/point_cloud_src_pcd_o3d.ply")
    print("Load a ply point cloud, print it, and render it")
    o3d.visualization.draw_geometries([pcd], zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])
