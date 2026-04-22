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
from src.tetris import Tetris
from src.model import Net
from utils.utils import Log, ela_t


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", type=bool, default=True)

    parser.add_argument("--render", type=bool, default=False, help='flag - video render')
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-2)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)

    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-2)

    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=1000,
                        help="Number of epochs between testing phases")

    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="trained_models")

    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--saved_path", type=str, default="weightFiles")

    args = parser.parse_args()
    return args


class Agent:
    def __init__(self, opt, device):
        self.epsilon = opt.initial_epsilon
        self.initial_epsilon, self.final_epsilon = opt.initial_epsilon, opt.final_epsilon
        self.epsilon_decay_step = opt.num_decay_epochs

        self.batch_size = opt.batch_size
        self.update_target_rate = 100

        self.replay_memory = deque(maxlen=opt.replay_memory_size)

        # generate model
        if opt.load_model:
            model = torch.load(opt.saved_path)
        else:
            model = Net(state_size=[opt.height,opt.width], out_size=1)
        self.main_q_network = model.to(device)
        self.target_q_network = model.to(device)
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
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))
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

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    agent = Agent(opt, device)

    optimizer = torch.optim.Adam(agent.main_q_network.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset().to(device)

    epoch = 0
    max_score, max_lines = 0, 0
    st = time()
    total_epoch = opt.num_epochs
    while epoch < total_epoch:
        next_steps = env.get_next_states()

        # Exploration or exploitation
        epsilon = agent.calc_epsilon(epoch)
        random_action = random() <= epsilon # random_action은 epsilon보다 작으면 True가 되어 탐험을 하게 됨

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
        reward, done = env.step(action, render=opt.render)

        agent.replay_memory.append([state, reward, next_state, done])   #  cache

        if done:
            final_score = env.score
            final_num_pieces = env.num_pieces
            final_cleared_lines = env.cleared_lines
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

        if final_cleared_lines > max_lines:
            max_score = final_score
            max_lines = final_cleared_lines
            torch.save(agent.main_q_network, f'{opt.saved_path}/tetris_E{epoch}_L{final_cleared_lines}')


        log_writer.write(
            f' Episode: {epoch:3d}/{opt.num_epochs}  |  Loss: {loss:6.3f}  |  Lines: {final_cleared_lines:2d} (max: {max_lines:2d}) '
            f' |  Score: {final_score:3d} (max: {max_score:2d})  |  ela: {ela_t(st)}'
        )

        loss.backward()
        optimizer.step()

        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Num Pieces', final_num_pieces, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)


if __name__ == "__main__":
    opt = get_args()
    train(opt)
