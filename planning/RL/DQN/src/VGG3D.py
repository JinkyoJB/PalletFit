import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np

class VGG3DSA(nn.Module):
    def __init__(self, input_shape, output_size, item_feat_dim=3):
        super(VGG3DSA, self).__init__()
        depth, height, width = input_shape  # (예: 20, 20, 20)
        
        self.features = nn.Sequential(
            # --- 3D Conv Blocks (생략) ---
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Conv 결과물의 Flatten 크기 계산
        self.flattened_size = self._get_flattened_size((1, depth, height, width))

        # 아이템 정보(item_feat_dim)까지 Concatenate된 크기 = flattened_size + item_feat_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size + item_feat_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
        )
        
    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
        return dummy_output.numel()

    def forward(self, bin_state, item_info):
        """
        bin_state : [batch, 1, depth, height, width]
        item_info : [batch, item_feat_dim]
        """
        x = self.features(bin_state)      # [batch, channels, d', h', w']
        x = x.view(x.size(0), -1)        # Flatten -> [batch, flattened_size]

        # 아이템 정보 Concatenate
        # => 최종 shape: [batch, flattened_size + item_feat_dim]
        x = torch.cat([x, item_info], dim=1)

        x = self.fc_layers(x)            # [batch, output_size]
        return x


class VGG3D(nn.Module):
    def __init__(self, input_shape, output_size):
        '''
        Q(s)를 근사하는 모델을 생성.
        '''
        super(VGG3D, self).__init__()
        depth, height, width = input_shape  # 입력 데이터 크기

        # VGG16과 유사한 Conv3D + MaxPool3D 블록
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 64, kernel_size=3, padding=1),  # [batch, 64, depth, height, width]
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),  # [batch, 64, depth, height, width]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [batch, 64, depth/2, height/2, width/2]

            # Block 2
            nn.Conv3d(64, 128, kernel_size=3, padding=1),  # [batch, 128, depth/2, height/2, width/2]
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),  # [batch, 128, depth/2, height/2, width/2]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [batch, 128, depth/4, height/4, width/4]

            # Block 3
            nn.Conv3d(128, 256, kernel_size=3, padding=1),  # [batch, 256, depth/4, height/4, width/4]
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),  # [batch, 256, depth/4, height/4, width/4]
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),  # [batch, 256, depth/4, height/4, width/4]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # [batch, 256, depth/8, height/8, width/8]

           # Block 4
           nn.Conv3d(256, 512, kernel_size=3, padding=1),  # [batch, 512, depth/8, height/8, width/8]
           nn.ReLU(),
           nn.Conv3d(512, 512, kernel_size=3, padding=1),  # [batch, 512, depth/8, height/8, width/8]
           nn.ReLU(),
           nn.Conv3d(512, 512, kernel_size=3, padding=1),  # [batch, 512, depth/8, height/8, width/8]
           nn.ReLU(),
           nn.MaxPool3d(kernel_size=2, stride=2),  # [batch, 512, depth/16, height/16, width/16]

            # # Block 5
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),  # [batch, 512, depth/16, height/16, width/16]
            # nn.ReLU(),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),  # [batch, 512, depth/16, height/16, width/16]
            # nn.ReLU(),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),  # [batch, 512, depth/16, height/16, width/16]
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=2, stride=2)  # [batch, 512, depth/32, height/32, width/32]
        )

        # Conv 결과의 크기를 계산
        flattened_size = self._get_flattened_size((1, depth, height, width))

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 64 * (input_shape[0] // 4) * (input_shape[1] // 4)),
            nn.ReLU(),
            nn.Linear(64 * (input_shape[0] // 4) * (input_shape[1] // 4), output_size)
        )

    def _get_flattened_size(self, input_shape):
        """
        Conv 레이어를 통과한 후의 데이터 크기를 계산합니다.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
        return dummy_output.numel()

    def forward(self, x):
        """
        전방 전달 정의
        """
        # Conv 연산
        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully Connected 연산
        x = self.fc_layers(x)
        return x

    def visualize_layer_io(self, state, save_dir="planning/RL/DQN/render/VGG3D"):
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
        for name, layer in self.features.named_modules():
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
    # depth, height, width = 20, 20, 20
    # output_size = 10
    # model = VGG3D(input_shape=(depth, height, width), output_size=output_size)

    # # 더미 데이터 생성
    # batch_size = 4
    # x = torch.randn(batch_size, 1, depth, height, width)
    # result = model(x)
    # print(f"Output shape: {result.shape}")

    # # 모델 전방 전달 (Hook을 통해 feature 저장)
    # model.visualize_layer_io(x)

    # ----------------------------------------
    depth, height, width = 20, 20, 20
    output_size = 10
    item_feat_dim = 3  # (depth, height, width) 등

    model = VGG3DSA(input_shape=(depth, height, width),
                  output_size=output_size,
                  item_feat_dim=item_feat_dim)

    # 더미 bin_state & item_info
    batch_size = 4
    bin_state = torch.randn(batch_size, 1, depth, height, width)
    item_info = torch.randn(batch_size, item_feat_dim)  # 예: (w,h,d)인 3차원 벡터
    print(f'item_info shape: {item_info.shape}')    # item_info shape: torch.Size([4, 3])
    print(f'item_info: {item_info}')  
    '''
    item_info: tensor([[-0.2157,  0.7020,  2.3088],
        [-0.5864, -1.0074, -0.0105],
        [ 0.2719,  1.9985, -0.6516],
        [-0.0414, -0.7864,  0.0950]])    
    '''  

    # Forward
    result = model(bin_state, item_info)
    print("Output shape:", result.shape)
    # -> [4, 1]