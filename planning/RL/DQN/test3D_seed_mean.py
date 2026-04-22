import argparse
import os
import torch
from src import Tetris3D, Net3D
import numpy as np

seed_list = [40]

def get_args(seed):
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to test Tetris3D""")
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--use_cuda", type=bool, default=True)

    parser.add_argument("--width", type=int, default=20, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--depth", type=int, default=20, help="The common depth for all images")

    parser.add_argument("--save_frame", type=bool, default=True)
    parser.add_argument("--convert_gif", type=bool, default=True)
    parser.add_argument("--fps", type=int, default=10, help="frames per second for GIF")

    weight_file = '$PALLETFIT_ROOT/planning/RL/DQN/weight/weightFiles3D_20250106-113131_202020_seed202020_2000_train_singleset/model_2000.pth'  # 저장된 모델 파일의 이름 (필요에 따라 변경)
    parser.add_argument("--saved_path", type=str, default=weight_file)
    weight_name = weight_file.split('/')[-2] +'_'+ weight_file.split('/')[-1]
    parser.add_argument("--out_images", type=str, default=f'test/{weight_name}_{seed}/')

    args = parser.parse_args()
    return args


def test(opt):
    os.makedirs(opt.out_images, exist_ok=True)  # 출력 이미지를 저장할 폴더 생성

    # 로그 파일 생성 및 열기
    log_file_path = os.path.join(opt.out_images, 'test_predicted.txt')
    with open(log_file_path, 'w') as log_file:

        # seed & device
        torch.manual_seed(opt.seed)
        device = 'cuda' if opt.use_cuda and torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.manual_seed_all(opt.seed)

        # 학습된 모델 로드
        # model = torch.load(opt.saved_path, map_location=device)
        # torch.serialization.add_safe_globals([model])
        # 모델을 초기화하고 가중치 로드
        model = Net3D(state_size=[opt.depth, opt.height, opt.width], out_size=1).to(device)
        # model = VGG3D(state_size=[opt.depth, opt.height, opt.width], out_size=1).to(device)
        model.load_state_dict(torch.load(opt.saved_path, map_location=device, weights_only=True))
        model.eval()  # 평가 모드로 설정

        if torch.cuda.is_available():
            model.cuda()

        # Tetris3D 환경 생성
        env = Tetris3D(width=opt.width, height=opt.height, depth=opt.depth, render=True, data_dir=opt.out_images, seed = seed)
        env.reset().to(device)

        while True:
            # 다음 상태와 액션을 예측
            next_steps = env.get_next_states()

            if len(next_steps) == 0:
                done = True
            
            else:
                next_actions, next_states = zip(*next_steps.items())
                next_states = torch.stack(next_states).to(device)

                with torch.no_grad():
                    predictions = model(next_states)[:, 0]


                    # 가장 높은 Q-value를 갖는 액션 선택
                    index = torch.argmax(predictions).item()
                    action = next_actions[index]

                    # 선택된 액션을 로그 파일에 기록

                    log_file.write(f'Item: {env.piece.shape}    |    Action: {action}\n')

                    # 선택된 액션으로 환경에서 한 스텝 수행
                    # reward, done = env.step(action)
                    reward, truncated, terminaled = env.step(action)
                    done = truncated or terminaled
                    env.render()

            # 에피소드가 끝나면 점수와 정보를 출력하고 로그 파일에 기록
            if done:
                score = env.score
                num_pieces = env.num_pieces
                lines_score = env.lines_score
                SU = env.get_space_utilization()
                log_file.write(f'Score: {score} | Cleared Lines: {lines_score} | Number of Pieces: {num_pieces}\n')
                print(f'Score: {score} | Cleared Lines: {lines_score} | Number of Pieces: {num_pieces} | Space Utilization: {SU:.4f}')
                break

        # 결과 후, 공간효율 계산
        space_efficiency = env.get_space_utilization()

        # 로그 파일에 공간 효율 기록
        log_file.write(f'Space Efficiency: {space_efficiency:.4f}\n')
        print(f'Space Efficiency: {space_efficiency:.4f}')
    
    env.save_gif(opt.out_images)
    return space_efficiency, env.num_pieces

if __name__ == "__main__":
    su_sum = 0
    piece_sum = 0
    su_list = []
    piece_list = []

    for seed in seed_list:
        opt = get_args(seed)
        su, pieces = test(opt)
        su_sum += su
        piece_sum += pieces
        su_list.append(su)
        piece_list.append(pieces)

    # 평균 계산
    su_mean = su_sum / len(seed_list) * 100  # 퍼센트로 변환
    pieces_mean = piece_sum / len(seed_list)

    # 표준편차 계산
    su_std = np.std(su_list) * 100  # 퍼센트로 변환
    pieces_std = np.std(piece_list)

    print(f"\nSummary:")
    print(f"Average Space Utilization: {su_mean:.2f} ± {su_std:.2f}")
    print(f"Average Number of Pieces: {pieces_mean:.2f} ± {pieces_std:.2f}")


