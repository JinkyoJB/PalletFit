import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

class Net3DSA(nn.Module):
    def __init__(self, state_size, out_size):
        super(Net3DSA, self).__init__()
        depth, height, width = state_size
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * (depth // 8) * (height // 8) * (width // 8), 512)
        self.fc2 = nn.Linear(512, out_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, depth/2, height/2, width/2]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, depth/4, height/4, width/4]
        x = self.pool(F.relu(self.conv3(x)))  # [batch, 128, depth/8, height/8, width/8]
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # [batch, out_size]
        return x

    def visualize_layer_io(self, state, save_dir="planning/RL/DQN/render/Net3DSA"):
        """
        Hook을 사용하여 Conv3D 레이어의 출력을 시각화.
        
        Args:
            state (torch.Tensor): 모델의 입력 상태, shape=[1, 1, depth, height, width].
            save_dir (str): 시각화를 저장할 디렉토리 경로.
        """
        os.makedirs(save_dir, exist_ok=True)

        # 입력을 GPU로 이동
        state = state.to(next(self.parameters()).device)

        # Hook을 통해 feature map 저장
        layer_outputs = {}
        def hook_fn(module, input, output):
            layer_outputs[module] = output.detach().cpu()

        # Conv3d 레이어에 Hook 등록
        hooks = []
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Conv3d):
                hooks.append(layer.register_forward_hook(hook_fn))

        # 모델 실행 (Forward Pass)
        with torch.no_grad():
            self.eval()
            self(state)

        # Hook 제거
        for hook in hooks:
            hook.remove()

        # 저장된 feature map 시각화
        for i, (layer, output) in enumerate(layer_outputs.items()):
            batch_size, channels, depth, height, width = output.shape
            print(f"Layer {i + 1}: {layer.__class__.__name__} - Shape: {output.shape}")

            # 첫 번째 배치의 첫 번째 채널만 시각화
            feature_map = output[0, 0, :, :, :].numpy()

            for d in range(min(3, depth)):  # Depth의 처음 3개만 시각화
                plt.imshow(feature_map[d, :, :], cmap="viridis")
                plt.title(f"Layer {i + 1} - Depth {d}")
                plt.colorbar()

                save_path = os.path.join(save_dir, f"layer_{i + 1}_depth_{d}.png")
                plt.savefig(save_path)
                plt.close()

        print(f"Feature maps saved to {save_dir}")

# Example usage
if __name__ == "__main__":
    state_size = (20, 20, 20)
    out_size = 800
    model = Net3DSA(state_size, out_size)

    # 더미 데이터 생성
    batch_size = 1
    x = torch.randn(batch_size, 1, *state_size)

    # 모델 실행 및 feature map 저장
    model.visualize_layer_io(x)

class Net3DS(nn.Module):
    def __init__(self, state_size, out_size=1):
        super(Net3DS, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=1),  # 커널 크기 축소
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),  # 패딩 조정
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        depth, height, width = state_size
        conv_output_size = self._get_conv_output_size(depth, height, width)

        self.fc = nn.Linear(conv_output_size, out_size)

    def _get_conv_output_size(self, depth, height, width):
        test_input = torch.zeros(1, 1, depth, height, width)
        out = self.conv0(test_input)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out.numel()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# class Net3D(nn.Module):
#     def __init__(self, state_size, out_size=2):
#         super(Net3D, self).__init__()
        
#         # Conv3d 레이어를 사용하여 3D 입력을 처리하도록 변경
#         # conv0: 초기 넓은 커널 크기(10x10x10)로 전체적인 큰 특징을 잡기 위한 레이어 추가
#         self.conv0 = nn.Sequential(
#             nn.Conv3d(1, 16, kernel_size=10, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2)
#         )
        
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2)
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2)
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv3d(64, 128, kernel_size=2, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2)
#         )
        
#         # 입력 차원은 Conv3d의 결과로 출력된 차원을 기반으로 계산
#         depth, height, width = state_size
#         conv_output_size = self._get_conv_output_size(depth, height, width)  # Conv3D 이후 출력 차원을 계산
        
#         # Fully Connected Layer
#         self.fc = nn.Linear(conv_output_size, out_size)

#         self._create_weights()

#     def _get_conv_output_size(self, depth, height, width):
#         # 테스트 입력 텐서를 이용해 Conv3D 이후의 출력 차원을 계산하는 함수
#         test_input = torch.zeros(1, 1, depth, height, width)  # 배치 크기 1, 채널 수 1, 입력 차원 (depth, height, width)
#         out = self.conv0(test_input)
#         out = self.conv1(out)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         return out.numel()  # 텐서의 전체 요소 수를 반환하여 FC 레이어의 입력 차원을 결정

#     def _create_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # Conv3d는 입력 차원을 [batch_size, channels, depth, height, width]로 기대
#         x = torch.unsqueeze(x, dim=1)  # channels dimension 추가 (예: [batch_size, 1, depth, height, width])
        
#         # 3D 컨볼루션 연산을 적용
#         out = self.conv0(x)
#         out = self.conv1(out)
#         out = self.conv2(out)
#         out = self.conv3(out)
        
#         # Fully Connected Layer를 위해 1차원으로 펼치기
#         out = out.view(out.size(0), -1)  # 텐서를 1차원으로 펼침
#         # 최종 출력
#         out = self.fc(out)
#         return out
    
# class Net3D(nn.Module):
#     def __init__(self, state_size, out_size=2):
#         super(Net3D, self).__init__()
        
#         # Conv3d 레이어를 사용하여 3D 입력을 처리하도록 변경
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(1, 32, kernel_size=5, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2)
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2)
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv3d(64, 128, kernel_size=2, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2)
#         )
        
#         # 입력 차원은 Conv3d의 결과로 출력된 차원을 기반으로 계산
#         depth, height, width = state_size
#         conv_output_size = self._get_conv_output_size(depth, height, width)  # Conv3D 이후 출력 차원을 계산
        
#         # Fully Connected Layer
#         self.fc = nn.Linear(conv_output_size, out_size)

#         self._create_weights()

#     def _get_conv_output_size(self, depth, height, width):
#         # 테스트 입력 텐서를 이용해 Conv3D 이후의 출력 차원을 계산하는 함수
#         test_input = torch.zeros(1, 1, depth, height, width)  # 배치 크기 1, 채널 수 1, 입력 차원 (depth, height, width)
#         out = self.conv1(test_input)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         return out.numel()  # 텐서의 전체 요소 수를 반환하여 FC 레이어의 입력 차원을 결정

#     def _create_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # Conv3d는 입력 차원을 [batch_size, channels, depth, height, width]로 기대
#         # 입력 텐서의 shape가 [batch_size, depth, height, width]인 경우
#         # 이를 [batch_size, channels, depth, height, width]로 확장
#         x = torch.unsqueeze(x, dim=1)  # channels dimension 추가 (예: [batch_size, 1, depth, height, width])
#         # print(x.shape)
#         # 3D 컨볼루션 연산을 적용
#         out = self.conv1(x)
#         # print(out.shape)
#         out = self.conv2(out)
#         # print(out.shape)
#         out = self.conv3(out)
#         # print(out.shape)
#         # Fully Connected Layer를 위해 1차원으로 펼치기
#         out = out.view(out.size(0), -1)  # 텐서를 1차원으로 펼침
#         # print(out.shape)
#         # 최종 출력
#         out = self.fc(out)
#         # print(out.shape)
#         return out

# # 모델 인스턴스 생성
# state_size = (13, 25, 25)  # 예시로 depth=32, height=32, width=32 사용
# model = Net3D(state_size=state_size, out_size=2)

# # TensorBoard SummaryWriter 설정
# log_dir = "logs3D/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = SummaryWriter(log_dir=log_dir)

# # 예제 입력 데이터 생성 (batch_size=1, depth=32, height=32, width=32)
# example_input = torch.rand(1, *state_size)

# # 모델 그래프 기록
# writer.add_graph(model, example_input)
# print(f"TensorBoard log is saved in: {log_dir}")

# # 기록 종료
# writer.close()
# test_input = torch.randn(1, 13, 25, 25)  # batch_size=1, depth=13, height=25, width=25
# output = model(test_input)

# # Example usage
# state_size = (20, 20, 20)  # depth, height, width
# model = Net3D(state_size=state_size, out_size=state_size[0]*state_size[1])
# test_input = torch.randn(1, *state_size)  # (batch_size, depth, height, width)
# output = model(test_input)
# print(output.shape)

# # Example usage
# state_size = (20, 20, 20)  # depth, height, width
# num_actions = state_size[0] * state_size[1] * 2  # (x, y, r) where r ∈ {0,1}

# # # Net3DSA
# model = Net3DSA(state_size=state_size, out_size=1).to('cuda' if torch.cuda.is_available() else 'cpu')
# test_input = torch.randn(20, 1, *state_size).to('cuda' if torch.cuda.is_available() else 'cpu')  # [1, 1, 20, 20, 20]
# output = model(test_input)
# print(f"Model Net3DSA output shape: {output.shape}")  # Should be [1, 800]

# # Net3DS
# model = Net3DS(state_size=state_size, out_size=1).to('cuda' if torch.cuda.is_available() else 'cpu')
# test_input = torch.randn(1, *state_size).to('cuda' if torch.cuda.is_available() else 'cpu')  # [1, 20, 20, 20]
# output = model(test_input)
# print(f"Model Net3DS output shape: {output.shape}")  # Should be [1, 1]
