# import torch.nn as nn
# import torch
# from torch.utils.tensorboard import SummaryWriter  # TensorBoard SummaryWriter 사용
# import datetime
# class Net(nn.Module):
#     def __init__(self, state_size, out_size=1):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.fc = nn.Linear(64 * (state_size[0]//4) * (state_size[1]//4), out_size)

#         self._create_weights()

#     def _create_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = torch.unsqueeze(x, dim=1)
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out


# if __name__ == '__main__':
#     # 모델 인스턴스 생성
#     state_size = (10, 10)  # 예제 입력 크기 (height, width)
#     model = Net(state_size=state_size, out_size=1)

#     # TensorBoard SummaryWriter 설정
#     log_dir = "logs/NetModel/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     writer = SummaryWriter(log_dir=log_dir)

#     # 예제 입력 데이터 생성 (batch_size=1, height=32, width=32)
#     example_input = torch.rand(1, *state_size)  # 예제 입력 크기 맞추기

#     # 모델 그래프 기록
#     writer.add_graph(model, example_input)
#     print(f"TensorBoard log is saved in: {log_dir}")

#     # 기록 종료
#     writer.close()

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class Net(nn.Module):
    def __init__(self, state_size, out_size=1):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64 * (state_size[0] // 4) * (state_size[1] // 4), out_size)

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x.shape = [batch, height, width]
        # Conv2d expects 4D: [batch, channel, H, W]
        x = torch.unsqueeze(x, dim=1)  # -> [batch, 1, height, width]
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def visualize_layer_io(self, state, save_dir="planning/RL/DQN/render/origin"):
        """
        Conv2D 레이어의 출력(feature map)을 Hook으로 수집하여 PNG 파일로만 저장.

        Args:
            state (torch.Tensor): 모델의 입력 상태, shape=[batch_size, height, width].
            save_dir (str): PNG 시각화를 저장할 디렉토리 경로.
        """
        os.makedirs(save_dir, exist_ok=True)

        # 디바이스
        device = next(self.parameters()).device
        state = state.to(device)

        # 레이어 출력 저장용 딕셔너리
        layer_outputs = {}

        def hook_fn(module, input, output):
            # Hook을 통해 Conv2d 레이어 출력을 CPU로 복사
            layer_outputs[module] = output.detach().cpu()

        # Conv2d 레이어에 Hook 등록
        hooks = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward Pass
        with torch.no_grad():
            self.eval()
            self(state)  # forward() 내부에서 [batch, 1, H, W]로 reshape

        # Hook 제거
        for h in hooks:
            h.remove()

        # 저장된 feature map 시각화
        for i, (layer, output) in enumerate(layer_outputs.items()):
            batch_size, channels, height, width = output.shape
            print(f"[Layer {i+1}] {layer.__class__.__name__} - output shape: {output.shape}")

            # 첫 번째 배치의 feature map
            # 채널이 많을 수 있으니 일부만 시각화
            for ch in range(min(channels, 3)):  # 3채널까지만
                feature_map_2d = output[0, ch].numpy()
                plt.imshow(feature_map_2d, cmap="viridis")
                plt.title(f"Layer {i+1} - Channel {ch}")
                plt.colorbar()

                # PNG 저장
                save_path = os.path.join(save_dir, f"layer_{i+1}_channel_{ch}.png")
                plt.savefig(save_path)
                plt.close()

        print(f"Feature maps saved to '{save_dir}'")

# ------------------ 예시 사용 ------------------
if __name__ == '__main__':
    # 모델 인스턴스 생성
    state_size = (10, 10)  # (height, width)
    model = Net(state_size=state_size, out_size=1)

    # 더미 입력 (batch_size=1, height=10, width=10)
    example_input = torch.rand(1, *state_size)

    # 시각화 함수 호출
    model.visualize_layer_io(example_input, save_dir="planning/RL/DQN/render/origin")

