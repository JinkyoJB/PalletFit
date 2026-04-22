import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

class Log:
    def __init__(self, log_dir):
        os.makedirs(log_dir) if not os.path.isdir(log_dir) else None
        self.log = open(log_dir+'/log.txt', 'wt')

    def write(self, s):
        print(s)
        self.log.write(s+'\n')

    def close(self):
        self.log.close()


def plot_3d_volume(data_3d, threshold=1e-6, title="3D Volume"):
    """
    data_3d: shape (D, H, W)인 3차원 텐서 또는 NumPy 배열
    threshold: 이 값 이상인 위치만 시각화 (ex: 0과 비교)
    """
    # 만약 torch.Tensor라면 NumPy로 변환
    if isinstance(data_3d, torch.Tensor):
        data_3d = data_3d.cpu().numpy()
    
    # data_3d.shape: (depth, height, width)
    depth, height, width = data_3d.shape
    
    # 0이 아닌 (또는 threshold 이상인) 위치 찾기
    # coords: (N, 3) -- [z, y, x]
    coords = np.argwhere(np.abs(data_3d) > threshold)
    if coords.size == 0:
        print("No voxels above threshold. Nothing to plot.")
        return
    
    # 해당 위치의 값 추출
    values = data_3d[coords[:, 0], coords[:, 1], coords[:, 2]]

    # 3D scatter
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # coords[:, 2] → width 방향 (x)
    # coords[:, 1] → height 방향 (y)
    # coords[:, 0] → depth 방향 (z)
    scatter = ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0],
                         c=values, cmap='viridis', marker='o', alpha=0.5)
    
    ax.set_xlabel("Width (x)")
    ax.set_ylabel("Height (y)")
    ax.set_zlabel("Depth (z)")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, depth)
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Value")
    
    plt.show(block=True)