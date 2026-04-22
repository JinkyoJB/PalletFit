from utils.constants import RotationType
import os
import torch
from RL.DQN.src import Tetris3D_origin
from RL.DQN.src import Net3DSA as Net3D
from RL.DQN.src import VGG3D
import numpy as np
import re
from collections import deque
from random import random, randint, sample
import time
from tensorboardX import SummaryWriter
from RL.DQN.utils.utils import Log, plot_3d_volume
import torch.nn as nn
import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib
matplotlib.use("TkAgg")
class Agent:
    def __init__(self, 
                 initial_epsilon,
                 final_epsilon,
                 num_decay_epochs,
                 batch_size,
                 update_target_rate,
                 replay_memory_size,
                 device,
                 width,
                 height,
                 depth,
                 load_weight=None,
                 ):
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.num_decay_epochs = num_decay_epochs
        self.batch_size = batch_size
        self.update_target_rate = update_target_rate
        self.width = width
        self.height = height
        self.depth = depth

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.device = device
        self.prev_params = None  # 이전 작업에서의 가중치 저장
        self.fisher_info = None  # Fisher 정보 저장
        # 네트워크 생성
        self.set_network(load_weight)

    def calc_epsilon(self, step):
        epsilon = self.final_epsilon + (max(self.num_decay_epochs - step, 0) * (
                self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
        return epsilon

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def set_network(self, weight_path):
        output_size = 1

        # 정규표현식으로 '_숫자_' 패턴을 찾음
        if weight_path is not None:
            match = re.search(r'_(\d+)_', weight_path)
            resolution_str = match.group(1)
            # 문자열을 두 자리씩 끊어서 정수 리스트로 변환
            self.resolution = [int(resolution_str[i:i+2]) for i in range(0, len(resolution_str), 2)]    # 2자리 넘기면 오류 있음.

            self.width = self.resolution[0]
            self.height = self.resolution[1]
            self.depth = self.resolution[2]

            # main_model = Net3D(state_size=[self.depth, self.height, self.width], out_size=output_size).to(self.device)
            main_model = VGG3D(input_shape=(self.depth, self.height, self.width), output_size=output_size).to(self.device)
            main_model.load_state_dict(torch.load(weight_path, map_location=self.device))
        else:
            # main_model = Net3D(state_size=[self.depth, self.height, self.width], out_size=output_size).to(self.device)
            main_model = VGG3D(input_shape=(self.depth, self.height, self.width), output_size=output_size).to(self.device)

        self.main_q_network = main_model

        # self.target_q_network = Net3D(state_size=[self.depth, self.height, self.width], out_size=output_size).to(self.device)
        self.target_q_network = VGG3D(input_shape=(self.depth, self.height, self.width), output_size=output_size).to(self.device)
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())
        self.target_q_network.eval()

    def visualize_weights(self, save_dir='planning/RL/DQN/render', layer_name='conv1'):
        """
        모델의 Conv3D 레이어 weight를 시각화하고 저장합니다.
        
        Args:
            save_dir (str): 이미지를 저장할 디렉토리 경로.
            layer_name (str): 시각화할 레이어 이름.
        """
        # 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)

        # Conv3D 레이어 weight 가져오기
        layer = dict(self.main_q_network.named_modules())[layer_name]
        weights = layer.weight.data.cpu().numpy()
        out_channels, in_channels, depth, height, width = weights.shape
        print(f"Visualizing weights for layer '{layer_name}':")
        print(f"Shape: {weights.shape}")

        # 각 채널의 weight를 시각화
        for out_channel in range(out_channels):
            for in_channel in range(in_channels):
                for d in range(depth):
                    # Weight 슬라이스 가져오기
                    weight_slice = weights[out_channel, in_channel, d]

                    # 시각화
                    plt.imshow(weight_slice, cmap='viridis')
                    plt.title(f'{layer_name} - out:{out_channel}, in:{in_channel}, depth:{d}')
                    plt.colorbar()

                    # 저장
                    filename = f'{layer_name}.png'
                    save_path = os.path.join(save_dir, filename)
                    plt.savefig(save_path)
                    plt.close()

    def visualize_layer_io(self, state, save_dir='planning/RL/DQN/render'):
        """
        Conv3d 레이어의 출력을 시각화합니다.
        
        Args:
            state (torch.Tensor): 모델의 입력 상태, shape=[1, 1, depth, height, width].
            save_dir (str): 시각화를 저장할 디렉토리 경로.
        """
        # 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)

        # 입력 데이터를 GPU로 이동
        state = state.to(self.device)

        # Hook 함수 정의
        layer_outputs = {}
        def hook_fn(module, input, output):
            layer_outputs[module] = output.detach().cpu()

        # Conv3d 레이어에 Hook 등록
        hooks = []
        for name, layer in self.main_q_network.named_modules():
            if isinstance(layer, torch.nn.Conv3d):
                hooks.append(layer.register_forward_hook(hook_fn))

        # 모델 실행 (Forward Pass)
        with torch.no_grad():
            self.main_q_network.eval()
            self.main_q_network(state)

        # Hook 제거
        for hook in hooks:
            hook.remove()

        # 레이어별 출력 시각화
        for i, (layer, output) in enumerate(layer_outputs.items()):
            output_array = output.squeeze().numpy()  # Remove batch dimension
            out_channels, depth, height, width = output_array.shape

            print(f"Layer {i + 1}: {layer.__class__.__name__} - Shape: {output_array.shape}")

            # 각 채널의 출력을 시각화
            for channel in range(out_channels):
                for d in range(depth):
                    slice_2d = output_array[channel, d, :, :]  # 각 채널의 특정 depth 슬라이스

                    plt.imshow(slice_2d, cmap='viridis')
                    plt.title(f'Layer {i + 1} Output - Channel {channel} - Depth {d}')
                    plt.colorbar()
                    save_path = os.path.join(save_dir, f'layer_{i + 1}_channel_{channel}_depth_{d}.png')
                    plt.savefig(save_path)
                    plt.close()

        print(f"Visualizations saved to {save_dir}")


    def compute_fisher_info(self):
        """
        Replay Memory를 기반으로 Fisher Information Matrix를 계산합니다.
        """
        if len(self.replay_memory) < self.batch_size:
            raise ValueError("Replay Memory가 충분하지 않습니다. Fisher 정보를 계산할 수 없습니다.")

        self.main_q_network.eval()
        fisher_info = {}
        for name, param in self.main_q_network.named_parameters():
            fisher_info[name] = torch.zeros_like(param)

        # Replay Memory에서 샘플링
        batch = sample(self.replay_memory, self.batch_size)
        state_batch, action_batch, _, _, _ = zip(*batch)
        state_batch = torch.cat(state_batch).to(self.device)

        for state in state_batch:
            state = state.unsqueeze(0)
            output = self.main_q_network(state)
            log_prob = torch.log_softmax(output, dim=1)
            prob = torch.softmax(output, dim=1)

            # Negative log likelihood 계산
            loss = -torch.sum(prob * log_prob)
            self.main_q_network.zero_grad()
            loss.backward()

            # Fisher 정보 업데이트
            for name, param in self.main_q_network.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.pow(2).detach()

        # Normalize Fisher 정보
        for name in fisher_info:
            fisher_info[name] /= len(state_batch)

        self.fisher_info = fisher_info
        self.prev_params = {name: param.clone().detach() for name, param in self.main_q_network.named_parameters()}

    def get_minibatch(self):
        """
        Retrieve a minibatch from replay memory, ensuring proper tensor formatting.
        """
        # 샘플링
        batch = sample(self.replay_memory, min(len(self.replay_memory), self.batch_size))

        # 데이터 분리
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # 텐서 변환
        state_batch = torch.cat(state_batch).to(self.device)  # [batch_size, 1, depth, height, width]
        action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device).unsqueeze(1)  # [batch_size, 1]
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)  # [batch_size, 1]
        next_state_batch = torch.cat(next_state_batch).to(self.device)  # [batch_size, 1, depth, height, width]
        done_batch = torch.tensor(done_batch, dtype=torch.bool, device=self.device)  # [batch_size]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch



class DQN:
    def __init__(self,  unfit_stop_setting = True, rotation_type = RotationType.BasicRotation, use_cuda = True):
        self.unfit_stop_setting = unfit_stop_setting
        self.rotation_type = rotation_type
        self.weight_path = "planning/RL/DQN/weight/weightFiles3D_20250208-033610_202020_20000_200_0/model_best.pth"
        # self.weight_path = None
        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu'

        self.resolution = [20, 20, 20]
        self.width = self.resolution[0]
        self.height = self.resolution[1]
        self.depth = self.resolution[2]

        self.steps = 20000
        self.decay_epoch = int(self.steps * 0.9)

        self.replay_memory_size = 800
        self.batch_size = 32
        self.initial_epsilon = 1
        self.final_epsilon = 0
        self.update_target_rate = 100

        self.gamma = 0.999
        self.save_interval = 2000
        self.ewc_lambda = 100  # EWC 정규화 강도

        self.agent = Agent(
            initial_epsilon=self.initial_epsilon,
            final_epsilon=self.final_epsilon,
            num_decay_epochs=self.decay_epoch,
            batch_size=self.batch_size,
            update_target_rate=self.update_target_rate,
            replay_memory_size=self.replay_memory_size,
            device=self.device,
            width=self.width,
            height=self.height,
            depth=self.depth,
            load_weight=self.weight_path,
        )

    def get_train_params(self):
        return {
            'width': self.width,
            'height': self.height,
            'depth': self.depth,
            'steps': self.steps,
            'decay_epoch': self.decay_epoch,
            'replay_memory_size': self.replay_memory_size,
            'batch_size': self.batch_size,
            'initial_epsilon': self.initial_epsilon,
            'final_epsilon': self.final_epsilon,
            'update_target_rate': self.update_target_rate,
            'gamma': self.gamma,
            'save_interval': self.save_interval,            
        }


    def train(self, target_bins, items_list, weight_path, nickname = 'revision_model', lr = 0.0005):

        self.agent.set_network(weight_path)

        time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        saved_path = f'planning/RL/DQN/weight/weightFiles3D_{time_stamp}_{self.width}{self.height}{self.depth}_{self.steps}_{nickname}'
        log_path = f'planning/RL/DQN/logs3D/{time_stamp}_{self.width}{self.height}{self.depth}_{self.steps}_{nickname}'

        os.makedirs(saved_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
        log_writer = Log(log_path)

        # ---- (1) 환경 생성 ----
        if len(target_bins) == 0:
            raise ValueError("No bins provided for DQN training.")
        bin_for_training = target_bins[0]  # 첫 번째 bin 사용

        # bin의 축척 계산
        origin_width = bin_for_training.width - (self.width - 1) * bin_for_training.margin_x
        origin_height = bin_for_training.height - (self.height - 1) * bin_for_training.margin_y
        origin_depth = bin_for_training.depth

        x_scale = origin_width / self.width
        y_scale = origin_height / self.height
        z_scale = origin_depth / self.depth

        env = Tetris3D_origin(
            width=self.width,
            height=self.height,
            depth=self.depth,
            render=False,
            mode='train'
        )
        self.action_space = env.action_space

        optimizer = torch.optim.Adam(self.agent.main_q_network.parameters(), lr=lr)
        
        # Cosine Annealing 스케줄러
        # scheduler = CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-6)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=1e-5)
        criterion = nn.MSELoss()

        step = 0  # epoch 대신 step 사용

        loss_history = []

        # 학습 옵션 저장
        log_writer.write(str(self.get_train_params()))
        log_writer.write(str(self.weight_path))

        w_2 = str(f'w_2: {env.w_2}')
        w_3 = str(f'w_3: {env.w_3}')
        w_4 = str(f'w_4: {env.w_4}')
        log_writer.write(w_2)
        log_writer.write(w_3)
        log_writer.write(w_4)

        for item in items_list:
            w, h, d = item.getDimension()
            w = np.ceil(w / x_scale)
            h = np.ceil(h / y_scale)
            d = np.ceil(d / z_scale)
            item_shape = [w, h, d]
            env.set_new_piece(item_shape)
        
        state = env.reset().unsqueeze(0).unsqueeze(1).to(self.device)  # [1,1,20,20,20]

        # ---- (2) 학습 시작 ----
        while step < self.steps:


            # print(f"Initial state shape: {state.shape}")  # 디버깅용 출력

            env.get_new_piece()
            env.num_pieces += 1

            next_states = env.get_next_states()
            # # next_states 시각화
            # for action, next_state in next_states.items():
            #     plot_3d_volume(next_state, title=f"next_state_{action}")

            # Exploration or exploitation
            epsilon = self.agent.calc_epsilon(step)
            is_random = random() <= epsilon

            if len(next_states) == 0:
                done = True
                state = env.reset().unsqueeze(0).unsqueeze(1).to(self.device)  # [1,1,20,20,20]
            else:
                next_actions, next_states = zip(*next_states.items())
                next_states = torch.stack(next_states).unsqueeze(1).to(self.device)  # [num_available,1,20,20,20]
                # print(f"Next states shape: {next_states.shape}")  # 디버깅용 출력

                self.agent.main_q_network.eval()
                with torch.no_grad():
                    # q_values = self.agent.main_q_network(next_states)  # [num_available, 2]
                    q_values = self.agent.main_q_network(next_states)[:,0]
                self.agent.main_q_network.train()

                if is_random:
                    selected_idx = randint(0, len(next_states) - 1)
                else:
                    # Available actions' indices in the full action space
                    # available_action_indices = [self.action_space.index(a) for a in next_actions]
                    # available_action_indices_tensor = torch.tensor(available_action_indices, device=self.device)

                    # # Gather Q-values for the available actions
                    # available_q_values = q_values[range(len(next_states)), available_action_indices_tensor]  # [num_available]
                    # print(f"Available action indices: {available_action_indices}")
                    # print(f"Available Q-values: {available_q_values}")

                    # Select the action with the highest Q-value
                    # selected_idx = torch.argmax(available_q_values).item()
                    # print(f"Selected action index: {selected_idx}")
                    
                    selected_idx = torch.argmax(q_values).item()    # Q-value가 가장 높은 action 선택


                action = next_actions[selected_idx]

                next_state = next_states[selected_idx].unsqueeze(0).to(self.device)  # [1,1,20,20,20]
                # print(f"Selected next state shape: {next_state.shape}")  # 디버깅용 출력
                reward, truncated, terminaled = env.step(action)
                done = truncated or terminaled
                state = torch.tensor(env.bin_matrix, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(1)

                # Replay Memory 저장: [state, action, reward, next_state, done]
                if selected_idx >= 0 and env.num_pieces > 2:
                    # print()
                    # unsqueeze_state = state.squeeze(0).squeeze(0)
                    # plot_3d_volume(unsqueeze_state)
                
                    self.agent.replay_memory.append([state, selected_idx, reward, next_state, done])
                    # print('Replay Memory:', len(self.agent.replay_memory))


            # Replay Memory가 충분하지 않으면 학습 스킵
            if len(self.agent.replay_memory) < self.replay_memory_size:
                log_writer.write(f'Saved Memory: {len(self.agent.replay_memory):4d}/{self.replay_memory_size} ({len(self.agent.replay_memory)/self.replay_memory_size*100:.1f}%)')
                continue

            if step % self.agent.update_target_rate == 0:
                self.agent.update_target_q_network()

            # step 업데이트
            step += 1

            # Mini-batch 학습
            if len(self.agent.replay_memory) >= self.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.agent.get_minibatch()

                # Q(s,a) 계산
                q_values = self.agent.main_q_network(state_batch)  # [batch_size, 6]
                # q_values = q_values.gather(1, action_batch)  # [batch_size, 1]

                # 타겟 Q(s,a) 계산
                # with torch.no_grad():
                #     next_q_values = self.agent.target_q_network(next_state_batch)  # [batch_size, 2]
                #     max_next_q_values, _ = next_q_values.max(dim=1, keepdim=True)  # [batch_size, 1]
                #     target_q_values = reward_batch + (self.gamma * max_next_q_values * (~done_batch).float().unsqueeze(1))  # [batch_size, 1]

                self.agent.main_q_network.eval()
                with torch.no_grad():
                    next_prediction_batch = self.agent.target_q_network(next_state_batch)

                self.agent.main_q_network.train()
                y_batch = torch.cat(
                    tuple(reward if done else reward + self.gamma * prediction
                        for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch))
                )[:, None]


                # 손실 함수 계산
                # loss = criterion(q_values, target_q_values)
                loss = criterion(q_values, y_batch)

                # # EWC 정규화 항 추가
                # if self.agent.fisher_info is not None:
                #     ewc_loss = 0
                #     for name, param in self.agent.main_q_network.named_parameters():
                #         if name in self.agent.fisher_info:
                #             fisher = self.agent.fisher_info[name]
                #             prev_param = self.agent.prev_params[name]
                #             ewc_loss += (fisher * (param - prev_param).pow(2)).sum()
                #     loss += self.ewc_lambda * ewc_loss  # λ로 조정

                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # log_writer.write(
                #     f"Step {step}, Loss: {loss.item():.4f}, SU: {final_SU:.4f}, Pieces: {final_num_pieces}, Score: {final_reward}, Time: {elapsed_time:.2f}s"
                # )
                # 손실 기록
                loss_history.append(loss.item())
                writer.add_scalar("Performance/Loss", loss.item(), step)
                writer.add_scalar("Performance/Epsilon", epsilon, step)
                writer.add_scalar("Performance/lr", scheduler.get_last_lr()[0], step)
                print(f"Step {step}, Loss: {loss.item()}")

                # if step == self.steps:
                #     state = state_batch[0].unsqueeze(0)
                    # self.agent.visualize_layer_io(state, save_dir=log_path+f'/{step}_layer')

                # 주기적으로 모델 저장
                if (step) % self.save_interval == 0 or step == self.steps:
                    torch.save(self.agent.main_q_network.state_dict(), f"{saved_path}/model_{step}.pth")


                # loss가 제일 작을 때 모델 저장
                if loss.item() == min(loss_history):
                    torch.save(self.agent.main_q_network.state_dict(), f"{saved_path}/model_best.pth")

                # # 최대 보상 갱신 시 모델 저장
                # if reward > max_reward:
                #     max_reward = reward
                #     torch.save(self.agent.main_q_network.state_dict(), f"{saved_path}/model_best.pth")

                # # ---- 로그 기록 및 TensorBoard 업데이트 ----
                if step % 2000 == 0:
                    # 학습된 최종 모델의 validation 결과 저장
                    val_env = self.stack(bin_for_training, items_list)
                    final_SU = val_env.get_space_utilization()
                    val_env.render(data_dir=saved_path, plot_name=f'final_result{step}_{final_SU:.2f}')
                    final_reward = val_env.reward
                    writer.add_scalar("Validation/SU", final_SU, step)
                    writer.add_scalar("Validation/reward", final_reward, step)

                # CUDA 메모리 사용량 기록
                if self.device == "cuda":
                    current_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    writer.add_scalar("Performance/CUDA Memory Allocated (MB)", current_allocated, step)

                # # 손실 안정성 계산
                # if len(loss_history) > window_size:
                #     recent_losses = loss_history[-window_size:]
                #     mean_loss = np.mean(recent_losses)
                #     std_loss = np.std(recent_losses)
                #     loss_stability = std_loss / mean_loss if mean_loss > 0 else 0
                #     print(f"Loss stability: {loss_stability}")



        writer.close()
        print("Training Done.")


    def stack(self,current_bin, current_item_list):

        # bin의 축척 계산
        origin_width = current_bin.width -(self.width -1)* current_bin.margin_x
        origin_height = current_bin.height -(self.height -1)* current_bin.margin_y
        origin_depth = current_bin.depth

        x_scale = origin_width / self.width
        y_scale = origin_height / self.height
        z_scale = origin_depth / self.depth
        
        # Tetris3D_origin 환경 생성
        self.env = Tetris3D_origin(width=self.width, height=self.height, depth=self.depth, render=False, mode = 'test')

        for idx, item in enumerate(current_item_list):
            w,h,d = item.getDimension()
            w = np.ceil(w / x_scale)
            h = np.ceil(h / y_scale)
            d = np.ceil(d / z_scale)
            item_shape = [w,h,d]
            self.env.set_new_piece(item_shape)

            if idx == 0:
                self.env.reset().to(self.device)

            self.env.get_new_piece()

            # 다음 상태와 액션을 예측
            next_states = self.env.get_next_states()

            # next_steps가 비어 있는 경우 종료 처리
            if not next_states:
                print("No valid next steps available. Ending the episode.")
                done = True
            else:
                next_actions, next_states = zip(*next_states.items())
                next_states = torch.stack(next_states).unsqueeze(1).to(self.device)
                
                self.agent.main_q_network.eval()
                # 모델을 사용하여 각 상태의 Q-value 예측
                with torch.no_grad():
                    q_values = self.agent.main_q_network(next_states)[:, 0]    # [num_next_actions, 2]
                
                selected_idx = torch.argmax(q_values).item()
                # available_action_indices = [self.action_to_index(a) for a in next_actions]
                # available_action_indices_tensor = torch.tensor(available_action_indices, device=self.device)

                # # Gather Q-values for the available actions
                # available_q_values = q_values[range(len(next_states)), available_action_indices_tensor]  # [num_available]
                # # print(f"Available action indices: {available_action_indices}")
                # # print(f"Available Q-values: {available_q_values}")

                # # Select the action with the highest Q-value
                # selected_idx = torch.argmax(available_q_values).item()
                # print(f"Selected action index: {selected_idx}")

                action = next_actions[selected_idx]

                print(f'Item: {self.env.piece.shape}    |    Action: {action}')

                # self.env.render()
                
                # 선택된 액션으로 환경에서 한 스텝 수행
                _, truncated, terminaled = self.env.step(action)
                current_pos = self.env.current_pos
                pivot = [current_pos['x']*x_scale, current_pos['y']*y_scale, current_pos['z']*z_scale]
                item.b_position = pivot
                if self.env.r == 1:
                    item.rotation_quat = 1
                current_bin.store(item)


                done = truncated or terminaled
            if done:
                return self.env
                # break


