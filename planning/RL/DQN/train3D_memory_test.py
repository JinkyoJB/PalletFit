import argparse
import torch
import torch.nn as nn
from random import sample
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src import Net3D, Tetris3D
from collections import deque
from mpl_toolkits.mplot3d import Axes3D

torch.cuda.empty_cache()  # Clear cache before starting

def get_args():
    parser = argparse.ArgumentParser("Implementation of Deep Q Network to play Tetris")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=40)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()
    return args

class Agent:
    def __init__(self, opt, device):
        self.replay_memory = deque(maxlen=opt.replay_memory_size)
        model = Net3D(state_size=[opt.depth, opt.height, opt.width], out_size=1)
        self.main_q_network = model.to(device)
        self.target_q_network = model.to(device)
        self.target_q_network.eval()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def get_minibatch(self):
        batch = sample(self.replay_memory, min(len(self.replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))
        return state_batch, reward_batch, next_state_batch, done_batch

def train_single_epoch(opt, device):
    env = Tetris3D(width=opt.width, height=opt.height, depth=opt.depth)
    agent = Agent(opt, device)
    optimizer = torch.optim.Adam(agent.main_q_network.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    # Perform a dummy forward pass to ensure memory allocation
    state = env.reset().to(device)
    next_steps = env.get_next_states()
    next_states = torch.stack([state.to(device) for _, state in next_steps.items()])
    with torch.no_grad():
        _ = agent.main_q_network(next_states)

    # Measure memory
    torch.cuda.synchronize()  # Ensure all operations are done before measuring
    memory_used = torch.cuda.memory_reserved(device)
    print(f"Memory used: {memory_used / (1024 ** 2):.2f} MB")

    # Clean up
    del agent, env, optimizer, criterion
    torch.cuda.empty_cache()
    return memory_used

if __name__ == "__main__":
    opt = get_args()

    num_decay_epochs_list = [100, 200, 500, 1000]
    replay_memory_size_list = [100, 200, 500, 1000]
    batch_size_list = [4, 8, 16, 32]
    update_target_rate_list = [50, 100, 200, 500]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        # GPU에 작은 텐서를 할당하여 메모리를 예약하도록 유도
        _ = torch.empty(1, device=device)

        # 현재 사용 가능한 메모리와 전체 메모리 확인
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = reserved_memory - allocated_memory

        print(f"총 GPU 메모리: {total_memory / (1024 ** 3):.2f} GB")
        print(f"예약된 GPU 메모리: {reserved_memory / (1024 ** 3):.2f} GB")
        print(f"사용 중인 GPU 메모리: {allocated_memory / (1024 ** 3):.2f} GB")
        print(f"사용 가능한 GPU 메모리: {free_memory / (1024 ** 3):.2f} GB")
    else:
        print("CUDA 사용 불가")    
    
    results = []

    for num_decay_epochs in num_decay_epochs_list:
        for replay_memory_size in replay_memory_size_list:
            for batch_size in batch_size_list:
                for update_target_rate in update_target_rate_list:
                    opt.num_decay_epochs = num_decay_epochs
                    opt.replay_memory_size = replay_memory_size
                    opt.batch_size = batch_size
                    opt.update_target_rate = update_target_rate
                    opt.lr = 0.0001
                    opt.load_model = False
                    opt.model_path = None

                    memory_used = train_single_epoch(opt, device)
                    results.append({
                        "num_decay_epochs": num_decay_epochs,
                        "replay_memory_size": replay_memory_size,
                        "batch_size": batch_size,
                        "update_target_rate": update_target_rate,
                        "memory_used_MB": memory_used / (1024 ** 2)
                    })

    df = pd.DataFrame(results)

    # Save results to a CSV file
    output_file = "results.csv"  # 파일 이름 설정
    df.to_csv(output_file, index=False)  # CSV 파일로 저장
    print(f"Results saved to {output_file}")


    # # heatmap_data = df.pivot_table(index='num_decay_epochs', columns='replay_memory_size', values='memory_used_MB')
    # plt.figure(figsize=(10, 8))
    # # sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Memory Usage (MB)'})
    # sns.pairplot(df, diag_kind="kde", kind="scatter", hue="memory_used_MB", palette="YlGnBu", plot_kws={"s": 60})

    # plt.title('Memory Usage Heatmap')
    # plt.xlabel('Replay Memory Size', fontsize=14)
    # plt.ylabel('Num Decay Epochs', fontsize=14)

    # # Increase tick label font size
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.show()

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Scatter plot with three variables from your DataFrame
    # ax.scatter(df['num_decay_epochs'], df['replay_memory_size'], df['memory_used_MB'], c=df['memory_used_MB'], cmap='YlGnBu', s=60)

    # # Set axis labels with increased font size
    # ax.set_xlabel('Num Decay Epochs', fontsize=14)
    # ax.set_ylabel('Replay Memory Size', fontsize=14)
    # ax.set_zlabel('Memory Usage (MB)', fontsize=14)

    # # Set tick label sizes
    # ax.tick_params(axis='both', which='major', labelsize=12)

    # plt.title('3D Memory Usage Analysis', fontsize=16)
    # plt.show()


