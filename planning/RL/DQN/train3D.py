import argparse
import os
import shutil
from random import random, randint, sample
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from collections import deque
from time import time
from src import Net3D, Tetris3D
from utils.utils import Log, ela_t
import datetime
 

# 모델 실행 이후 메모리 비우기
torch.cuda.empty_cache()

num_epoch = 50000
#decay_epoch는 num_epoch의 90%로 설정
decay_epoch = int(num_epoch * 0.9)
replay_memory_size = 100
batch_size = 16
window_size = 10
width = 10
height = 10
depth = 10

render = False

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda", type=bool, default=True)

    parser.add_argument("--width", type=int, default=width, help="The common width for all images")
    parser.add_argument("--height", type=int, default=height, help="The common height for all images")
    parser.add_argument("--depth", type=int, default=depth, help="The common depth for all images")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=0)

    parser.add_argument("--num_decay_epochs", type=float, default=decay_epoch)

    parser.add_argument("--num_epochs", type=int, default=num_epoch)
    parser.add_argument("--batch_size", type=int, default=batch_size, help="The number of images per batch")
    parser.add_argument("--replay_memory_size", type=int, default=replay_memory_size, help="Number of epochs between testing phases")

    parser.add_argument("--lr", type=float, default=0.00125)

    parser.add_argument("--save_interval", type=int, default=1000)

    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="planning/RL/DQN/weight/weightFiles3D_20241220-084810_202020_replay100_batch16_safety_epsilon0/model_50000.pth")
    
    nickname = f'{width}{height}{depth}_replay{replay_memory_size}_batch{batch_size}_safety_epsilon0_coding'
    parser.add_argument("--log_path", type=str, default=f"planning/RL/DQN/logs3D/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{nickname}")
    parser.add_argument("--saved_path", type=str, default=f"planning/RL/DQN/weight/weightFiles3D_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{nickname}")
    parser.add_argument("--render", type=bool, default=render, help='flag - render')


    args = parser.parse_args()
    return args


class Agent:
    def __init__(self, opt, device):
        self.epsilon = opt.initial_epsilon
        self.initial_epsilon, self.final_epsilon = opt.initial_epsilon, opt.final_epsilon
        self.epsilon_decay_step = opt.num_decay_epochs

        self.batch_size = opt.batch_size
        self.update_target_rate = 10

        self.replay_memory = deque(maxlen=opt.replay_memory_size)
        self.device = device

        # generate model
        if opt.load_model:
            # model = torch.load(opt.model_path)
            model = Net3D(state_size=[opt.depth, opt.height, opt.width], out_size=1).to(self.device)
            # model = VGG3D(state_size=[opt.depth, opt.height, opt.width], out_size=1).to(self.device)
            model.load_state_dict(torch.load(opt.model_path, map_location=self.device))
            print(f'Loaded model from {opt.model_path}')
        else:
            model = Net3D(state_size=[opt.depth, opt.height, opt.width], out_size=1)
            # model = VGG3D(state_size=[opt.depth, opt.height, opt.width], out_size=1)
        self.main_q_network = model.to(self.device)
        self.target_q_network = model.to(self.device)
        self.target_q_network.eval()
        self.update_target_q_network()

    def calc_epsilon(self, epoch):
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        return epsilon

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def get_minibatch(self):
        batch = sample(self.replay_memory, min(len(self.replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))
        done_batch = torch.tensor(done_batch, dtype=torch.bool, device=self.device)  # GPU에 직접 올리기
        return state_batch, reward_batch, next_state_batch, done_batch


def train(opt):
    # seed & device
    torch.manual_seed(opt.seed)
    device = 'cuda' if opt.use_cuda and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(opt.seed)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)

    os.makedirs(opt.log_path)
    os.makedirs(opt.saved_path) if not os.path.isdir(opt.saved_path) else None
    writer = SummaryWriter(opt.log_path)
    log_writer = Log(opt.log_path)

    # env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env = Tetris3D(width=opt.width, height=opt.height, depth=opt.depth, seed=opt.seed, render=opt.render)
    agent = Agent(opt, device)

    optimizer = torch.optim.Adam(agent.main_q_network.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset().to(device)
    max_reward = 0
    epoch = 0
    total_epoch = opt.num_epochs
    start_time = time.time()
    loss_stability = 0


    if 'loss_history' not in globals():
        loss_history = []

    # 학습 옵션 저장
    log_writer.write("Training complete. Saving all options:")
    for key, value in vars(opt).items():
        log_writer.write(f"{key}: {value}")

    print(f'opt.batch_size: {opt.batch_size}, epoch: {opt.num_epochs}, opt.gamma: {opt.gamma}, opt.lr: {opt.lr}')


    while epoch < total_epoch:
        # print(f'epoch: {epoch}')
        next_steps = env.get_next_states()

        # Exploration or exploitation
        epsilon = agent.calc_epsilon(epoch)
        random_action = random() <= epsilon

        if len(next_steps) == 0:
            done = True
        
        else:
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states).to(device)

            agent.main_q_network.eval()
            with torch.no_grad():
                predictions = agent.main_q_network(next_states)[:, 0]
            agent.main_q_network.train()
            if random_action:
                index = randint(0, len(next_steps) - 1)   # 랜덤하게 action 선택, 탐험
            else:
                index = torch.argmax(predictions).item()    # Q-value가 가장 높은 action 선택

            next_state = next_states[index, :]
            action = next_actions[index]
            reward, truncated, terminaled = env.step(action)
            done = truncated or terminaled


        agent.replay_memory.append([state, reward, next_state, done])   #  cache

        if done:
            final_score = env.score
            final_num_pieces = env.num_pieces
            final_lines_score = env.lines_score
            final_SU = env.get_space_utilization()
            final_safety_score = env.safety_score
            elapsed_time = time() - start_time

            state = env.reset().to(device)

        else:
            state = next_state
            continue

        if len(agent.replay_memory) < opt.replay_memory_size/10:
            log_writer.write(f'Saved Memory: {len(agent.replay_memory):4d}/{opt.replay_memory_size} ({len(agent.replay_memory)/opt.replay_memory_size*100:.1f}%)')
            continue

        if epoch % agent.update_target_rate == 0:
            agent.update_target_q_network()

        epoch += 1
        state_batch, reward_batch, next_state_batch, done_batch = agent.get_minibatch()
        state_batch, reward_batch, next_state_batch\
            = state_batch.to(device), reward_batch.to(device), next_state_batch.to(device)

        q_values = agent.main_q_network(state_batch)

        agent.main_q_network.eval()
        with torch.no_grad():
            next_prediction_batch = agent.target_q_network(next_state_batch)

        agent.main_q_network.train()
        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction
                  for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch))
        )[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)

        loss_history.append(loss.item())

        if (epoch+1) % opt.save_interval == 0 or epoch == total_epoch - 1 :
            torch.save(agent.main_q_network.state_dict(), f"{opt.saved_path}/model_{epoch+1}.pth")

        if reward > max_reward:
            max_reward = reward
            torch.save(agent.main_q_network.state_dict(), f"{opt.saved_path}/model_best.pth")

        # 현재 CUDA 메모리 사용량 기록
        if device == "cuda":
            current_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB 단위
            current_reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB 단위
            writer.add_scalar("Performance/CUDA Memory Allocated (MB)", current_allocated, epoch)
            writer.add_scalar("Performance/CUDA Memory Reserved (MB)", current_reserved, epoch)
            # log_writer.write(
            #     f"Epoch {epoch}, CUDA memory allocated: {current_allocated:.2f} MB, reserved: {current_reserved:.2f} MB"
            # )

        # 매 epoch 종료 시 추가 기록
        if len(loss_history) > window_size:  # window_size: 최근 n개의 loss를 고려
            recent_losses = loss_history[-window_size:]
            mean_loss = np.mean(recent_losses)
            std_loss = np.std(recent_losses)

            # Loss Stability: 표준 편차 / 평균
            loss_stability = std_loss / mean_loss if mean_loss > 0 else 0

        # 로그 기록
        log_writer.write(
            f"Epoch {epoch}, Loss: {loss:.4f}, SU: {final_SU:.4f}, Pieces: {final_num_pieces},  Score: {final_score}, Time: {elapsed_time:.2f}s"
        )

        loss.backward()
        optimizer.step()

        # TensorBoard 기록
        writer.add_scalar("Performance/Loss", loss, epoch)
        writer.add_scalar("Performance/Loss Stability", loss_stability, epoch)
        writer.add_scalar("Performance/Epsilon", epsilon, epoch)
        writer.add_scalar("Performance/Elapsed Time", elapsed_time, epoch)

        writer.add_scalar("Performance/Space Utilization", final_SU, epoch)
        writer.add_scalar("Performance/Number of Pieces", final_num_pieces, epoch)
        writer.add_scalar("Performance/Line Score", final_lines_score, epoch)
        writer.add_scalar("Performance/Safety Score", final_safety_score, epoch)
        writer.add_scalar("Performance/score", final_score, epoch)





if __name__ == "__main__":
    opt = get_args()
    train(opt)
