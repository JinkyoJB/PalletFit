import numpy as np  # 배열 및 수학적 연산을 위한 라이브러리
from PIL import Image  # 이미지를 다루기 위한 PIL(Pillow) 라이브러리
import cv2  # OpenCV: 이미지 및 영상 처리 라이브러리
from matplotlib import style  # 그래프 스타일 설정 라이브러리
import torch  # PyTorch: 머신러닝 및 딥러닝 연산을 위한 라이브러리
import random  # 랜덤 값 생성을 위한 라이브러리
from datetime import datetime  # 현재 날짜 및 시간 정보를 다루는 라이브러리
import matplotlib.pyplot as plt  # 그래프 및 시각화를 위한 Matplotlib 라이브러리
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 3D 시각화를 위한 라이브러리

style.use('ggplot')  # Matplotlib의 그래프 스타일을 'ggplot'으로 설정

# 테트리스에 사용될 각 테트로미노 조각의 정보를 정의한 클래스
class Tetromino:
    '''
    테트리스에 쓰일 조각에 대한 정보가 담긴 클래스
    '''
    def __init__(self):
        # 테트로미노 조각 ID를 정의
        self.pieces = [1, 2, 3, 4, 5, 6, 7]  # 조각의 ID (1~7)
        
        # 각 테트로미노 조각의 색상(RGB) 정의
        colors = [         # RGB
            [0, 0, 0],     # background (검은색)
            [240, 55, 55], # 빨간색
            [110, 200, 55],# 초록색
            [250, 180, 55],# 주황색
            [250, 170, 170],# 분홍색
            [170, 170, 250],# 연파란색
            [50, 80, 250], # 파란색
            [50, 200, 250],# 청록색
        ]

        # 각 테트로미노 조각의 2D 배열 형태를 정의
        pieces_detail = [
            [],

            [[1, 1],
             [1, 1]],  # 정사각형 모양 조각

            [[0, 2, 0],
             [2, 2, 2]],  # T자형 조각

            [[0, 3, 3],
             [3, 3, 0]],  # Z자형 조각

            [[4, 4, 0],
             [0, 4, 4]],  # S자형 조각

            [[5, 5, 5, 5]],  # I자형 조각

            [[0, 0, 6],
             [6, 6, 6]],  # J자형 조각

            [[7, 0, 0],
             [7, 7, 7]]  # L자형 조각
        ]

        # 색상을 BGR 순서로 변환하고, numpy 배열로 변환하여 저장
        self.colors = [np.array(c[::-1], dtype=np.uint8) for c in colors]

        # 테트로미노 조각의 형태를 numpy 배열로 변환하여 저장
        self.pieces_detail = [np.array(p, dtype=np.uint8) for p in pieces_detail]

    def __len__(self):
        # 테트로미노 조각의 개수를 반환
        return len(self.pieces)

    def __getitem__(self, idx):
        # 특정 인덱스에 해당하는 테트로미노 조각의 형태를 반환
        return self.pieces_detail[idx]

    def random_choice(self):
        # 임의의 테트로미노 조각을 선택하여 반환
        piece_id = random.choice(self.pieces)
        return piece_id, self.pieces_detail[piece_id]
    

class Tetromino:
    '''
    테트리스에 쓰일 조각에 대한 정보가 담긴 클래스
    '''
    def __init__(self):
        # 테트로미노 조각 ID를 정의
        self.pieces = [1, 2, 3, 4, 5, 6, 7]  # 조각의 ID (1~7)
        
        # 각 테트로미노 조각의 색상(RGB) 정의
        colors = [         # RGB
            [0, 0, 0],     # background (검은색)
            [240, 55, 55], # 빨간색
            [110, 200, 55],# 초록색
            [250, 180, 55],# 주황색
            [250, 170, 170],# 분홍색
            [170, 170, 250],# 연파란색
            [50, 80, 250], # 파란색
            [50, 200, 250],# 청록색
        ]

        # 각 테트로미노 조각의 2D 배열 형태를 정의
        pieces_detail = [
            [],

            [[1, 1],
             [1, 1]],  # 정사각형 모양 조각

            [[0, 2, 0],
             [2, 2, 2]],  # T자형 조각

            [[0, 3, 3],
             [3, 3, 0]],  # Z자형 조각

            [[4, 4, 0],
             [0, 4, 4]],  # S자형 조각

            [[5, 5, 5, 5]],  # I자형 조각

            [[0, 0, 6],
             [6, 6, 6]],  # J자형 조각

            [[7, 0, 0],
             [7, 7, 7]]  # L자형 조각
        ]

        # 색상을 BGR 순서로 변환하고, numpy 배열로 변환하여 저장
        self.colors = [np.array(c[::-1], dtype=np.uint8) for c in colors]

        # 테트로미노 조각의 형태를 numpy 배열로 변환하여 저장
        self.pieces_detail = [np.array(p, dtype=np.uint8) for p in pieces_detail]

    def __len__(self):
        # 테트로미노 조각의 개수를 반환
        return len(self.pieces)

    def __getitem__(self, idx):
        # 특정 인덱스에 해당하는 테트로미노 조각의 형태를 반환
        return self.pieces_detail[idx]

    def random_choice(self):
        # 임의의 테트로미노 조각을 선택하여 반환
        piece_id = random.choice(self.pieces)
        return piece_id, self.pieces_detail[piece_id]


class Tetris:
    tetromino = Tetromino() # 큐브를 클래스변수로 생성

    def __init__(self, height=20, width=10, block_size=20, out_images=None):
        # 테트리스 보드의 높이, 너비, 블록 크기 등을 설정
        self.height, self.width, self.block_size = height, width, block_size
        self.board_size = (height, width)  # 보드의 크기 저장
        
        # 게임 정보가 표시될 보드의 크기와 배경색 설정
        h, w, c = (self.block_size*5), (self.width * self.block_size), 3
        bg_color = (255, 255, 255)  # 배경색: 흰색
        self.text_color = (0, 0, 0)  # 텍스트 색상: 검정색
        self.info_board = np.ones((h, w, c), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        
        # 출력할 이미지 경로 설정
        self.out_images = out_images
        
        # 게임 초기화
        self.reset()

    def reset(self):
        # 게임 보드 초기화: 0으로 채워진 배열
        self.board = np.zeros(self.board_size, dtype=np.uint8)

        # 게임 점수, 조각 개수, 제거된 줄 수 초기화
        self.score = 0
        self.num_pieces = 0
        self.cleared_lines = 0

        # 임의의 테트로미노 조각을 선택하여 저장
        self.piece_id, self.piece = self.tetromino.random_choice()
        piece_h, piece_w = self.piece.shape  # 선택된 조각의 높이와 너비

        # 조각의 초기 위치 설정 (보드 상단 중앙)
        self.current_pos = {
            'x': self.width//2 - piece_w//2,  # 보드 중앙
            'y': 0  # 상단에서 시작
        }

        # 게임 오버 상태 초기화
        self.gameover = False
        return self.get_board_state(self.board)

    def rotate(self, piece):
        # 조각을 90도 회전하여 반환
        return np.rot90(piece)

    def get_board_state(self, board):
        # 현재 보드의 상태를 이진화하여 텐서로 반환
        state = (board != 0).astype(int)
        return torch.FloatTensor(state)

    def get_next_states(self):
        # 현재 조각의 다음 가능한 상태들을 반환
        states = {}  # 가능한 상태를 저장할 딕셔너리
        num_rotations = 0  # 가능한 회전 횟수를 초기화

        # 조각의 ID에 따라 가능한 회전 횟수를 설정
        if self.piece_id == 1:  # 정사각형 조각은 회전 필요 없음
            num_rotations = 1
        elif self.piece_id in [3, 4, 5]:  # Z자형, S자형, I자형 조각은 2번 회전 가능
            num_rotations = 2
        elif self.piece_id in [2, 6, 7]:  # T자형, J자형, L자형 조각은 4번 회전 가능
            num_rotations = 4

        curr_piece = self.piece.copy()  # 현재 조각 복사
        for i in range(num_rotations):
            # 가로 위치를 조각이 보드 밖으로 벗어나지 않도록 제한
            valid_xs = self.width - curr_piece.shape[1]
            for x in range(valid_xs+1):
                piece = curr_piece.copy()
                pos = {'x': x, 'y': 0}  # 조각의 초기 위치 설정

                # 조각이 바닥에 닿거나 충돌할 때까지 y 위치를 증가시킴
                while not self.check_collision(piece, pos):
                    pos['y'] += 1

                # 충돌이 발생하면 조각을 잘라내고, 보드에 저장
                _, piece = self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_board_state(board)  # 상태를 저장
            curr_piece = self.rotate(curr_piece)  # 조각을 회전
        return states

    def get_current_board_state(self):
        # 현재 보드 상태를 반환 (현재 조각을 포함한 상태)
        board = self.board.copy()
        h, w = self.piece.shape
        board[self.current_pos['y']:self.current_pos['y']+h, self.current_pos['x']:self.current_pos['x']+w] += self.piece
        return board

    def new_piece(self):
        # 새로운 테트로미노 조각을 생성하여 초기 위치에 배치
        self.piece_id, self.piece = self.tetromino.random_choice()
        piece_h, piece_w = self.piece.shape
        self.current_pos = {
            'x': self.width//2 - piece_w//2,  # 중앙에서 시작
            'y': 0  # 상단에서 시작
        }

        # 만약 새로운 조각이 배치되었을 때 충돌이 발생하면 게임 오버로 설정
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        # 조각이 현재 위치에서 보드의 다른 블록과 충돌하는지 확인
        future_y = pos['y'] + 1  # 한 칸 아래로 이동한 y 좌표
        h, w = piece.shape
        # 보드의 현재 위치에 조각이 배치될 수 있는지 확인 (충돌 여부)
        board_status = (self.board[future_y:future_y+h, pos['x']:pos['x']+w] != 0).astype(int)
        if board_status.shape != piece.shape:
            return True  # 크기가 맞지 않으면 충돌
        overlap = (board_status*2) - np.where(piece>1, 1, piece) == 1
        if np.sum(overlap) > 0 or np.sum(np.array(range(h))+future_y > self.height-1) > 0:
            return True  # 겹치거나 보드를 벗어나면 충돌로 판단
        return False

    def truncate(self, piece, pos):
        # 조각이 보드 상단을 벗어날 때, 조각의 상단을 잘라내고 게임 오버를 설정
        def get_last_collision_row(h, board_status, piece):
            overlap = (board_status * 2) - np.where(piece > 1, 1, piece) == 1
            tmp = np.where(np.sum(overlap, axis=-1) >= 1, 1, np.sum(overlap, axis=-1))
            last_collision_row = h - (np.argmax(tmp[::-1]) + 1) if np.sum(tmp) != 0 else -1
            return last_collision_row

        gameover = False
        h, w = piece.shape
        board_status = (self.board[pos['y']:pos['y']+h, pos['x']:pos['x']+w] != 0).astype(int)
        last_collision_row = get_last_collision_row(h, board_status, piece)

        if pos['y'] - (h - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and piece.shape[0] > 1:
                gameover = True
                piece = piece[1:, :]  # 조각의 상단 한 줄을 제거
                board_status = board_status[:piece.shape[0], :]
                last_collision_row = get_last_collision_row(piece.shape[0], board_status, piece)
        return gameover, piece

    def store(self, piece, pos):
        # 현재 조각을 보드에 저장
        board = self.board.copy()
        h, w = piece.shape
        board_status = (board[pos['y']:pos['y']+h, pos['x']:pos['x']+w] != 0).astype(int)
        overlap = (board_status*2) - np.where(piece>1, 1, piece) == 1
        if np.sum(overlap) == 0:
            board[pos['y']:pos['y']+h, pos['x']:pos['x']+w] += piece
        return board

    def check_cleared_rows(self, board):
        # 보드에서 꽉 찬 줄을 찾아 제거
        to_delete = np.where(np.sum((board != 0).astype(int), axis=-1) == self.width)[0]
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        # 보드에서 특정 줄을 제거하고 위의 줄들을 아래로 이동
        for i in indices:
            board = np.concatenate((np.zeros((1, self.width), dtype=np.uint8), board[:i], board[i+1:]))
        return board

    def step(self, action, render=True, save_frame=None):
        # 하나의 행동을 수행하여 보드의 상태를 업데이트
        x, num_rotations = action  # x 위치와 회전 수
        self.current_pos = {"x": x, "y": 0}  # 조각의 초기 위치 설정
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)  # 조각 회전

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1  # 충돌이 없을 때까지 조각을 아래로 이동
            if render:
                self.render(save_frame)  # 조각 이동 시 화면에 렌더링

        # 충돌이 발생하면 조각의 충돌된 부분을 자르고 보드에 배치
        overflow, piece = self.truncate(self.piece, self.current_pos)
        self.board = self.store(piece, self.current_pos)
        if overflow:  # 충돌이 발생하여 조각이 잘려나갔으면 게임 오버
            self.gameover = True
            if render:
                self.render(save_frame, done=True)

        # 꽉 찬 줄 제거 및 점수 업데이트
        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.num_pieces += 1
        self.cleared_lines += lines_cleared
        if not self.gameover:
            self.new_piece()  # 새로운 조각 생성
        if self.gameover:
            self.score -= 2  # 게임 오버 시 점수 감소

        return score, self.gameover

    def render(self, save_frame=None, done=False):
        # 현재 게임 보드를 화면에 렌더링
        if not self.gameover:
            img = np.expand_dims(self.get_current_board_state(), axis=-1)
            for piece_id in self.tetromino.pieces:
                img = np.where(img == piece_id, self.tetromino.colors[piece_id], img)
        else:
            img = np.expand_dims(self.board, axis=-1)
            for piece_id in self.tetromino.pieces:
                img = np.where(img == piece_id, self.tetromino.colors[piece_id], img)

        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        img = img.resize((self.width * self.block_size, self.height * self.block_size), 0)
        img = np.array(img)
        img[[i * self.block_size for i in range(self.height)], :, :] = 0
        img[:, [i * self.block_size for i in range(self.width)], :] = 0

        img = np.concatenate((self.info_board, img), axis=0)

        cv2.putText(img, "Score:", (self.block_size, self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)
        cv2.putText(img, str(self.score),
                    (7 * self.block_size, self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)

        cv2.putText(img, "N Pieces:", (self.block_size, 2 * self.block_size+int(self.block_size / 2)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)
        cv2.putText(img, str(self.num_pieces),
                    (7 * self.block_size, 2 * self.block_size + int(self.block_size / 2)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)

        cv2.putText(img, "Lines:", (self.block_size, 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)
        cv2.putText(img, str(self.cleared_lines),
                    (7 * self.block_size, 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)

        if save_frame:
            if done:
                img = np.array(Image.fromarray(img, "RGB").convert('L'))
            cv2.imwrite(f"{self.out_images}/{datetime.now().strftime('%H_%M_%S_%f')}.jpg", img)

        cv2.imshow("DQN Tetris", img)
        cv2.waitKey(1)


# 메인 함수: 테트리스 게임 실행
if __name__ == '__main__':     
    height, width, block_size = 18, 10, 20  # 보드의 높이, 너비, 블록 크기 설정

    env = Tetris(height, width, block_size)  # 테트리스 환경 초기화

    # 게임 루프
    while True:
        # 임의의 액션 선택 (x 위치, 회전 수)
        action = [random.choice(range(width-4)), 0]  # range(width-4) = range(0, 6), action = [x 좌표, 회전 수]
        _, done = env.step(action, render=True, save_frame=True)  # 선택한 액션 수행 및 보드 업데이트

        if done:  # 게임 오버 시 루프 종료
            break
