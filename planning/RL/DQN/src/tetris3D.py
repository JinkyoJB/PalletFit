import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import imageio
from collections import deque
# from RL.DQN.utils.utils import Log, plot_3d_volume

import matplotlib
# matplotlib.use("TkAgg")   # planning 내부용, GUI모드에서는 주석처리필요


class Tetromino3D:
    '''
    사용될 큐브의 종류와 색상을 정의하는 클래스
    '''
    def __init__(self):
        self.default_colors = [
            [0, 0, 0],      # 배경색 (ID: 0)
            [169, 169, 169] # 탐색색 (ID: 255)만 우선 있다고 가정
        ]
        self.colors = [np.array(color, dtype=np.uint8) for color in self.default_colors]

        self.tetromino_type_list = []
        self.pieces = []
        # pieces_detail[0]은 비워두는 관례
        self.pieces_detail = [[]]  
        # # 각 큐브의 형태 정의
        # self.pieces_detail = [
        #     [],
        #     np.array([[
        #             [1, 2],
        #             [3, 4]],
        #             [
        #             [5, 6],
        #             [7, 8]]]),  # rotation 확인용 2x2x2 큐브
        #     np.ones((1, 1, 1), dtype=np.uint8),  # 1x1x1 큐브
        #     np.ones((2, 2, 2), dtype=np.uint8),  # 2x2x2 큐브
        #     np.ones((3, 3, 3), dtype=np.uint8),  # 3x3x3 큐브

        # ]

    def set_tetromino_type(self, tetromino_type_list=None):
        """
        새로운 tetromino_type_list를 받아서 pieces, pieces_detail 등을 재설정하는 메서드.
        만약 인자가 None이면 self.default_tetromino_type_list 사용.
        """
        if tetromino_type_list is None:
            default_tetromino_type_list = [
                        [4, 2, 2],
                        [4, 3, 2],
                        [3, 3, 3],
                        [3, 2, 3],
                        [3, 5, 3]
                    ]
            tetromino_type_list = default_tetromino_type_list

                    # (1) 풀 색상 목록: _original_default_colors에 보관
            _original_default_colors = [
                [0, 0, 0],       # 배경색 (ID: 0)
                [240, 55, 55],   # 빨간색 (ID: 1)
                [110, 200, 55],  # 초록색 (ID: 2)
                [50, 80, 250],   # 파란색 (ID: 3)
                [240, 240, 55],  # 노란색 (ID: 4)
                [240, 160, 55],  # 주황색 (ID: 5)
                [169, 169, 169]  # 회색 (탐색 중인 조각 - ID: 255)
            ]
            self.default_colors = _original_default_colors[:]
            self.colors = [np.array(color, dtype=np.uint8) for color in self.default_colors]

        # 1) tetromino_type_list 갱신
        self.tetromino_type_list = tetromino_type_list
        
        # 2) pieces (1부터 시작)
        self.pieces = list(range(1, len(tetromino_type_list) + 1))
        
        # 3) pieces_detail 구성
        new_details = [np.ones((np.array(item).astype(int)), dtype=np.uint8)
                       for item in tetromino_type_list]
        new_details.insert(0, [])  # 0번 인덱스 비우기
        self.pieces_detail = new_details
        
        # 4) 만약 colors가 부족하다면 → 확장
        needed_colors = len(self.pieces)
        # 0번, 255번 제외한 실제 개수
        existing_colors = len(self.default_colors) - 1  
        if needed_colors > existing_colors:
            extra_colors_needed = needed_colors - existing_colors
            for _ in range(extra_colors_needed):
                self.default_colors.insert(
                    -1,
                    [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                )
            self.colors = [np.array(color, dtype=np.uint8) for color in self.default_colors]
        
        print("[INFO] Tetromino types, pieces, and details have been updated.")

    def add_tetromino_type(self, new_shape):
        """
        예: new_shape = [5,5,5] 와 같이 전달받으면
        기존 tetromino_type_list에 추가하고, 
        pieces, pieces_detail, (필요 시) colors를 업데이트한다.
        """
        # 1) 기존 목록에 추가
        self.tetromino_type_list.append(new_shape)

        # 2) ID 설정
        new_id = max(self.pieces) + 1 if self.pieces else 1
        self.pieces.append(new_id)
        
        # 3) pieces_detail 확장
        new_piece_detail = np.ones((np.array(new_shape).astype(int)), dtype=np.uint8)
        self.pieces_detail.append(new_piece_detail)
        
        # 4) colors 부족 시 확장
        needed_colors = len(self.pieces)
        existing_colors = len(self.default_colors) - 1
        if needed_colors > existing_colors:
            extra_colors_needed = needed_colors - existing_colors
            for _ in range(extra_colors_needed):
                self.default_colors.insert(
                    -1,
                    [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                )
        self.colors = [np.array(color, dtype=np.uint8) for color in self.default_colors]
        
        return new_id

    def random_choice(self):
        """pieces 내 임의의 큐브 ID를 뽑아서 반환"""
        piece_id = random.choice(self.pieces)
        return piece_id, self.pieces_detail[piece_id]

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        return self.pieces_detail[idx]

    def has_same_shape(self, tetromino_shape):
        """
        tetromino_shape = [3,3,3] 과 같이 주어지면,
        self.pieces_detail 중에서 이 shape와 동일한 배열이 있는지 확인해서
        True/False를 반환한다.
        """
        # 리스트 [3,3,3] → 튜플 (3,3,3)로 변환
        target_shape = tuple(tetromino_shape)
        
        # pieces_detail[0]은 관례상 비어있으므로, 실제 조각은 1번 인덱스부터 확인
        for detail in self.pieces_detail[1:]:
            if isinstance(detail, np.ndarray) and detail.shape == target_shape:
                return True
        return False
    
    def get_piece_id_by_shape(self, tetromino_shape):
        """
        tetromino_shape = [3,3,3] 처럼 주어지면,
        self.pieces_detail 중에서 이 shape와 동일한 배열을 가진 조각의 ID를 반환한다.
        만약 찾지 못하면 None을 반환한다.
        """
        target_shape = tuple(tetromino_shape)
        
        # self.pieces에 들어있는 ID 순회
        for piece_id in self.pieces:
            detail = self.pieces_detail[piece_id]  # 0번 인덱스는 비어 있으므로 piece_id는 1 이상
            if isinstance(detail, np.ndarray) and detail.shape == target_shape:
                return piece_id
        
        return None  # 동일한 shape를 가진 조각이 없는 경우


class Tetris3D:
    def __init__(self, width=10, height=20, depth=10, render=False, mode = 'train'):
        self.width = width
        self.height = height
        self.depth = depth
        self.bin_matrix_size = (depth, height, width)  # z, y, x 순서
        self.tetromino3d = Tetromino3D() # 클래스 변수
        self.mode = mode
        if self.mode == 'random':
            self.tetromino3d.set_tetromino_type()
        # 출력할 이미지 경로 설정
        self.render = render # render할 지 여부
        self.frames = []  # 각 프레임을 저장할 리스트

        '''
        w1_score = self.w_1 * self.lines_score
        w2_score = self.w_2 * self.SU
        w3_score = self.w_3 * self.safety_score
        w4_score = self.w_4 * self.weight_distritbution
        w5_score = self.w_5 * self.origin_proximity
        '''
        # self.w_1 = 0
        self.w_2 = 0
        self.w_3 = 0
        self.w_4 = 1

        self.tetromino_list = deque()
        self.action_space = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]

        # self.reset()

    def reset(self):
        # 게임 상태 초기화
        self.bin_matrix = np.zeros(self.bin_matrix_size, dtype=np.uint8)

        self.reward = 0  # 평가 점수
        self.num_pieces = 1 # 평가 element2
        self.lines_score = 0  # 평가 element3, 기존의 cleared_lines 확장
        self.SU = 0
        self.safety_score = 0
        self.weight_distritbution = 0
        self.last_item = {'item': [0, 0, 0], 'position': {'x': 0, 'y':0, 'z': 0}}   # item: [d,h,w]

        if self.mode == 'random':
            self.get_new_piece()

        return self.get_board_state(self.bin_matrix)
    
    def get_direction(self, x, piece_h, piece_w):
        direction = {
            0: (-piece_w, 0),   # 좌측
            1: (self.last_item['item'][2], 0),   # 우측
            2: (0, -piece_h),   # 뒤
            3: (0, self.last_item['item'][1]),   # 앞
            4: (0, 0),                           # 위 (x, y 좌표는 그대로)
        }
        return direction[x]
    
    def get_item_info(self):
        return torch.tensor([self.piece_d, self.piece_h, self.piece_w], dtype=torch.float32)
    
    def rotate(self, piece):
        # print(f'Rotating piece: {self.piece_id}')
        return np.rot90(piece, axes=(1, 2)) #numpy.rot90(m, k=1, axes=(0, 1)) m: 회전시킬 배열, k: 회전 횟수(1:90), axes: 회전할 축

    def get_board_state(self, bin_matrix):
        state = (bin_matrix != 0).astype(int)
        return torch.FloatTensor(state)
    
    def test_bin_matrix(self):
        # 테스트를 위해 self.bin_matrix의 1층을 1로 채움
        self.bin_matrix[0, :, :] = 1
    
    def get_next_states(self):
        states = {}

        for i in self.action_space:
            x, r = i
            piece = self.tetromino3d[self.piece_id]
            piece = np.transpose(piece, (2, 1, 0))

            if r == 1:
                piece = self.rotate(piece)

            piece_d, piece_h, piece_w = piece.shape

            if self.last_item['item'] == [0, 0, 0]:
                pos = {'x': 0, 'y': 0, 'z': 0}

            else:
                dx, dy = self.get_direction(x, piece_h, piece_w)
                pos = {
                    'x': self.last_item['position']['x'] + dx,
                    'y': self.last_item['position']['y'] + dy,
                    'z': self.depth - piece_d
                }

                while not self.check_collision(piece, pos):
                    pos['z'] -= 1
                
                truncated, terminaled = self.check_terminal(piece, pos, r)

                if truncated:
                    continue
            bin_matrix = self.store(piece, pos)
            item_info = torch.tensor([piece_d, piece_h, piece_w])
            # self.plot_piece(bin_matrix, plot_name='get_next_states')
            states[(x, r)] = [self.get_board_state(bin_matrix), item_info]   # i: 회전 방향
        
        return states
       
    def get_current_board_state(self):
        bin_matrix = self.bin_matrix.copy()
        d,h,w = self.piece.shape

        # 현재 조각을 탐색 중인 조각으로 표시하기 위해 255로 설정
        bin_matrix[self.current_pos['z']:self.current_pos['z']+d,
              self.current_pos['y']:self.current_pos['y']+h,
              self.current_pos['x']:self.current_pos['x']+w] = 255

        return bin_matrix

    def set_new_piece(self, piece_shape):
        # self.tetromino3D에 piece_shape가 있는지 확인
        id = self.tetromino3d.get_piece_id_by_shape(piece_shape)
        if id is None:
            id = self.tetromino3d.add_tetromino_type(piece_shape)
        
        self.tetromino_list.append(id)
        return id


    def get_new_piece(self):
        if self.mode == 'random':
            '''
            임의의 조각을 선택하여 piece_id, piece, piece_h, piece_w, piece_d, current_pos를 설정
            '''
            self.piece_id, self.piece = self.tetromino3d.random_choice()

        try:
            self. piece_id = self.tetromino_list.popleft()
            self.piece = self.tetromino3d.pieces_detail[self.piece_id]
            self.piece = np.transpose(self.piece, (2, 1, 0))  # z, y, x 순서로 맞추기 위해 transpose
            self.piece_d, self.piece_h, self.piece_w = self.piece.shape
            self.current_pos = {'x': 0, 'y': 0, 'z': self.depth - self.piece_d}
            return True
        except:
            return False


    
    def check_collision(self, piece, pos):
        # print(f'check_collision: {pos}')
        future_z = pos['z'] - 1
        depth, height, width = piece.shape  # 조각의 깊이, 높이, 너비를 가져옴

        # board_status의 축을 piece와 동일하게 맞춤 (z, y, x로 재배열)
        board_status = (self.bin_matrix[
            future_z:future_z + depth,
            pos['y']:pos['y'] + height,
            pos['x']:pos['x'] + width
        ] != 0).astype(int)

        if board_status.shape != piece.shape:
            return True  # 크기가 맞지 않으면 bin_matrix에서 벗어난 것으로 판단

        overlap = (board_status * 2) - np.where(piece > 1, 1, piece) == 1
        if np.sum(overlap) > 0 or np.sum(np.array(range(depth)) + future_z > self.depth - 1) > 0:
            return True  # 겹치면 충돌로 판단

        return False
    
    def check_terminal(self, piece, pos, rotation):
        '''
        episode의 종료를 판단하는 함수.: search_space가 비어있을 때 종료됨. 겹치는 위치가 있다면 search_space에서 제거
        '''

        

        search_space = self.action_space.copy()
        depth, height, width  = piece.shape
        board_status = (self.bin_matrix[
            pos['z']:pos['z']+depth,
            pos['y']:pos['y']+height,
            pos['x']:pos['x']+width
        ] != 0).astype(int)
        truncated = False   # 배치 불가능, episode 종료
        terminaled = True   # 배치 가능, episode 종료

        # pos의 요소 중 하나라도 음수이면 truncated, terminaled = True
        if pos['x'] < 0 or pos['y'] < 0 or pos['z'] < 0:
            truncated = True
            terminaled = True
            return truncated, terminaled

        # 가로,세로,높이 중 하나라도 범위를 벗어나는지 확인
        if pos['x'] + width > self.width or pos['y'] + height > self.height or pos['z'] + depth > self.depth:
            truncated = True
            terminaled = True
            return truncated, terminaled
        

        # board_status와 piece의 겹치는 부분을 찾기 위해 겹치는 영역을 계산
        overlap = (board_status * 2) - np.where(piece > 1, 1, piece) == 1

        # 겹치는 부분이 있는지 확인
        if np.sum(overlap) > 0:
            truncated = True

            # 겹치는 부분의 좌표 찾기
            overlap_positions = np.argwhere(overlap)
            overlap_positions = overlap_positions + [pos['z'], pos['y'], pos['x']]

            # x 범위에서 overlap_positions이 포함되는 요소 제거
            max_overlap_x = overlap_positions[:, 2].max()
            pos_y = pos['y']
            for i in range(pos['x'], max_overlap_x +1 ):
                if (i, pos_y, rotation) in search_space:
                    search_space.remove((i, pos_y, rotation))

            # y 범위에서 overlap_positions이 포함되는 요소 제거
            max_overlap_y = overlap_positions[:, 1].max()
            pos_x = pos['x']
            for j in range(pos['y'], max_overlap_y +1 ):
                if (pos_x, j, rotation) in search_space:
                    search_space.remove((pos_x, j, rotation))

        else:
            # print('겹치는 부분 없음, 해당 위치에 저장됨')
            terminaled = False

        if len(search_space) == 0:
            terminaled = True

        return truncated, terminaled

    def store(self, piece, pos):
        bin_matrix = self.bin_matrix.copy()
        depth, height, width = piece.shape
        bin_matrix[pos['z']:pos['z']+depth, pos['y']:pos['y']+height, pos['x']:pos['x']+width] = self.piece_id
        return bin_matrix
    

    def calcul_lines(self):
        """
        각 층(높이 slice)마다 가로와 세로에서 연속된 1의 최댓값을 구해 점수를 계산하는 함수.
        state의 shape는 (높이, 행, 열)로 가정.
        """
        def longest_run(arr):
            """
            1차원 배열에서 연속된 1의 최대 개수를 반환하는 함수.
            """
            max_run = 0
            current_run = 0
            for item in arr:
                if item == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            return max_run
        # 예를 들어, self.get_board_state(self.bin_matrix)는 (20,20,20) 배열을 반환한다고 가정
        state = np.array(self.get_board_state(self.bin_matrix))
        total_score = 0

        # 각 층(높이)에 대해 계산
        for level in state:  # level의 shape: (20, 20)
            # 가로 방향: 각 행별로 연속된 1의 최대 개수를 구한 뒤, 그 중 최댓값을 선택
            max_horizontal = max(longest_run(row) for row in level)
            
            # 세로 방향: 각 열별로 연속된 1의 최대 개수를 구한 뒤, 그 중 최댓값을 선택
            # level.T를 사용하면 열이 행으로 전환됩니다.
            max_vertical = max(longest_run(col) for col in level.T)

            # print(f'level: {level.shape}, max_horizontal: {max_horizontal}, max_vertical: {max_vertical}')
            
            # 해당 층의 점수는 가로와 세로의 최댓값의 합
            total_score += max_horizontal**2 + max_vertical**2

        return total_score
    

    # def calcul_origin_proximity(self):
        
    #     return 
    
    def calcul_weight_distribution(self):
        """
        state의 무게분포를 기반으로 안전성 점수를 계산하는 함수.
        
        각 층에 존재하는 블록(값이 1인 셀)에 대해 해당 층의 가중치를 곱하여 전체 안전성 점수를 구합니다.
        예를 들어, 높이가 20인 보드라면, 
        - 가장 위 층은 가중치 1,
        - 가장 아래 층은 가중치 20을 부여합니다.
        
        반환되는 값은 전체 블록의 가중 합을 전체 블록 수로 나눈 '평균 가중치'로, 
        값이 클수록 블록이 아래쪽(안정적인 위치)에 집중되어 있다고 판단할 수 있습니다.
        """
        # 현재 보드 상태 얻기 (예: shape (20, 20, 20))
        state = np.array(self.get_board_state(self.bin_matrix))
        height, _, _ = state.shape

        # 각 층에 가중치 부여: 1부터 height까지 (아래층일수록 높은 가중치)
        # 예: 층 0 -> 1, 층 1 -> 2, ..., 층 height-1 -> height
        weights = np.arange(height, 0, -1).reshape(height, 1, 1)
        
        # 각 셀의 안전 기여도: 해당 셀의 값(0 또는 1) * 그 층의 가중치
        weighted_sum = np.sum(state * weights)
        
        # 전체 블록 수 (state에서 1인 셀의 개수)
        total_blocks = np.sum(state)
        
        # 블록이 없으면 안전성 점수를 0으로 처리
        if total_blocks == 0:
            return 0
        
        return weighted_sum
        
        # # 안전성 점수를 평균 가중치로 계산
        # safety_score = weighted_sum / total_blocks
        # return safety_score
    
    def calcul_safety(self):
        '''
        현재 step에서 배치한 아이템의 하중지지점수를 계산하여 self.safety_score에 저장

        1. 아이템의 아래 셀이 모두 채워지는 경우 +1
        1층인 경우 +2
        2. 아이템의 아랫면쪽 4개의 꼭지점이 채워져 있는 경우 0
        3. 아이템의 아랫면쪽 꼭지점이 채워져 있지 않은 경우 -1 * (채워져 있지 않은 꼭지점의 개수)
        '''
        if self.current_pos['z'] == 0:
            return 10
        if self.bin_matrix[self.current_pos['z'] - 1, self.current_pos['y'], self.current_pos['x']] != 0:
            return 1
        
        # 아이템의 각 꼭지점이 채워져 있는지 확인
        # 꼭지점셀의 좌표
        corners = [
            (self.current_pos['z']-1, self.current_pos['y'], self.current_pos['x']), # 왼쪽 아래, (4, 3, 0)
            (self.current_pos['z']-1, self.current_pos['y'] + self.piece_h -1 , self.current_pos['x']), # 왼쪽 위, (-1, 4, 0)
            (self.current_pos['z']-1, self.current_pos['y'], self.current_pos['x'] + self.piece_w -1), # 오른쪽 아래, (-1, 3, 3)
            (self.current_pos['z']-1, self.current_pos['y'] + self.piece_h-1, self.current_pos['x'] + self.piece_w -1)   # 오른쪽 위, (-1, 4, 3)
        ]
        corner_filled = [self.bin_matrix[z, y, x] != 0 for z, y, x in corners]
        if all(corner_filled):
            return 0
        else:
            return - 1
        
    
    def search_outermost(self, bin_matrix):
        """
        bin_matrix에서 '1이 아닌(≠1) 셀'들로 구성된 각 연결 덩어리를 찾고,
        각 덩어리의 3D Bounding Box 꼭지점(8개 이하)을 반환합니다.
        
        Returns:
            corners_list (list): 각 덩어리에 대한 꼭지점 좌표들의 리스트.
            예) [
                [ (z1, y1, x1), (z1, y1, x2), ..., (z2, y2, x2) ],  # 1번째 덩어리 8개 모서리
                [ (z3, y3, x3), ...],                              # 2번째 덩어리 ...
                ...
            ]
        """
        # 1) '1이 아닌' 셀들만 추출 → 이 좌표들을 통해 연결된 영역을 탐색
        mask = (bin_matrix != 0)
        depth, height, width = bin_matrix.shape

        visited = np.zeros_like(bin_matrix, dtype=bool)

        # 방향 (6방향: 위/아래/앞/뒤/좌/우)
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if mask[z, y, x] and not visited[z, y, x]:
                        # (z,y,x)가 '1이 아닌' 셀이며, 아직 방문 안했다면 → 새 덩어리 시작
                        queue = deque()
                        queue.append((z, y, x))
                        visited[z, y, x] = True

                        # 이 덩어리에 속하는 좌표들의 min/max 추적
                        z_min, z_max = z, z
                        y_min, y_max = y, y
                        x_min, x_max = x, x

                        while queue:
                            cz, cy, cx = queue.popleft()
                            # min/max 갱신
                            if cz < z_min: z_min = cz
                            if cz > z_max: z_max = cz
                            if cy < y_min: y_min = cy
                            if cy > y_max: y_max = cy
                            if cx < x_min: x_min = cx
                            if cx > x_max: x_max = cx

                            # 이웃 탐색
                            for dz, dy, dx in directions:
                                nz, ny, nx = cz+dz, cy+dy, cx+dx
                                if (0 <= nz < depth and
                                    0 <= ny < height and
                                    0 <= nx < width):
                                    if mask[nz, ny, nx] and not visited[nz, ny, nx]:
                                        visited[nz, ny, nx] = True
                                        queue.append((nz, ny, nx))

                        # BFS/DFS 끝난 후, (z_min, z_max, y_min, y_max, x_min, x_max) 얻음
                        # 이로부터 3D bounding box의 8개 꼭지점(모서리)을 만들 수 있음
                        # corner (z0, y0, x0), z0 in [z_min, z_max], y0 in [y_min, y_max], x0 in [x_min, x_max]
                        # 단, z_min==z_max 등으로 겹치면 꼭지점이 8개 미만이 될 수도 있음
                        cpoints = []
                        for zz in [z_min, z_max]:
                            for yy in [y_min, y_max]:
                                for xx in [x_min, x_max]:
                                    cpoints.append((zz, yy, xx))


        return cpoints
    
    def get_space_utilization(self):
        total_cells = self.width * self.height * self.depth
        empty_cells = np.sum(self.bin_matrix == 0)
        return 1 - (empty_cells / total_cells)
    
    def step(self, action):
        x, r = action
        self.r = r  # self.stack 전송용

        self.piece = self.tetromino3d[self.piece_id]
        self.piece = np.transpose(self.piece, (2, 1, 0))  # z, y, x 순서로 맞추기 위해 transpose
        if r == 1:
            self.piece = self.rotate(self.piece)
        self.piece_d, self.piece_h, self.piece_w = self.piece.shape
        # ─────────────────────────────────────────────────────────
        # (2) bin이 비어 있으면 (0,0,0)에 배치하고 드롭 로직 스킵
        if not np.any(self.bin_matrix):
            # bin_matrix에 0이 아닌 값이 하나도 없는 상태 → 완전 빈 상태
            self.current_pos = {'x': 0, 'y': 0, 'z': 0}
        else:
            # 일반적인(기존) 경우에는 맨 위(depth - piece_d)에서부터 드롭
            # self.last_item의 position과 x를 기준으로 현재 위치 계산
            '''
            action[0]: 이전아이템의 좌측(0), 우측(1), 뒤(2), 앞(3), 위(4) 위치
            action[1]: 0이면 회전안함, 1이면 z축기준 90도 회전
            '''
            outermost = self.search_outermost(self.bin_matrix)
            outermost_w = outermost[-1][2] + 1
            outermost_h = outermost[-1][1] + 1
            outermost_d = outermost[-1][0] + 1

            outermost_pos = {'x': outermost[0][2], 'y': outermost[0][1], 'z': outermost[0][0]}
            
            self.last_item = {'item': [outermost_d, outermost_h, outermost_w], 'position': outermost_pos }
                
            dx, dy = self.get_direction(x, piece_h=self.piece_h, piece_w=self.piece_w)

            self.current_pos = {
                'x': self.last_item['position']['x'] + dx,
                'y': self.last_item['position']['y'] + dy,
                'z': self.depth - self.piece_d
                }

            while not self.check_collision(self.piece, self.current_pos):
                self.current_pos['z'] -= 1
        # ─────────────────────────────────────────────────────────

        truncated, terminaled = self.check_terminal(self.piece, self.current_pos, r)
        # self.plot_piece(plot_name=f'train3D_check, x:{action[0]}, y:{action[1]}  size:{self.piece.shape} \n truncated:{truncated}, terminaled:{terminaled}')


        if truncated:
            # self.lines_score = self.calcul_lines()
            # self.SU = self.get_space_utilization()
            # self.safety_score = self.calcul_safety()
            # self.weight_distritbution = self.calcul_weight_distribution()
            # w1_score = self.w_1 * self.lines_score
            # w2_score = self.w_2 * self.SU
            # w3_score = self.w_3 * self.safety_score
            # w4_score = self.w_4 * self.weight_distritbution
            # print(f'truncated: w1_score:{w1_score:.2f}, w2_score:{w2_score:.2f}, w3_score:{w3_score:.2f}, SU:{self.SU:.2f}', f'weight_distritbution:{self.weight_distritbution:.2f}')
            # self.reward = w1_score + w2_score + w3_score + w4_score  # 한 줄이 가득 차면 1점 추가 , 1번 case
            print(f'truncated: SU:{self.SU:.2f}')
            self.reward  = -1 
            # self.render()
            return self.reward , truncated, terminaled

        else:
            self.bin_matrix = self.store(self.piece, self.current_pos)
            self.last_item = {'item': [self.piece_d, self.piece_h, self.piece_w], 'position': self.current_pos}

            self.lines_score = self.calcul_lines()
            self.SU = self.get_space_utilization()
            self.safety_score = self.calcul_safety()
            self.weight_distritbution = self.calcul_weight_distribution()

            # self.plot_piece(plot_name = 'not truncated')

            # self.reward = self.w_1 * self.lines_score + self.w_2 * self.SU  # 한 줄이 가득 차면 1점 추가 , 2번 case
            # w1_score = self.w_1 * self.lines_score
            w2_score = self.w_2 * self.SU
            w3_score = self.w_3 * self.safety_score
            w4_score = self.w_4 * self.weight_distritbution
            # print(f'not truncated: w1_score:{w1_score:.2f}, w2_score:{w2_score:.2f}, w3_score:{w3_score:.2f}, SU:{self.SU:.2f}', f'weight_distritbution:{self.weight_distritbution:.2f}')
            # self.reward = w1_score + w2_score + w3_score + w4_score # 한 줄이 가득 차면 1점 추가 , 1번 case
            print(f'not truncated: w3(safety_score):{w3_score:.2f}, w4(weight_distritbution):{self.weight_distritbution:.2f} :: SU:{self.SU:.2f}')
            self.reward = w2_score + w3_score + w4_score # 한 줄이 가득 차면 1점 추가 , 1번 case
            
            # self.render(plot_name ='in step')

            if self.mode == 'random':
                if not terminaled:
                    self.get_new_piece()
                    self.num_pieces += 1

            return self.reward , truncated, terminaled
        
    
    def render(self, several_axis_view=False, plot_name=None, data_dir='planning/renders'):
        # self.data_dir 디렉토리가 없으면 생성
        if data_dir:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
        image_path = self.plot_piece(plot_name = plot_name, several_axis_view = several_axis_view, data_dir=data_dir)
        self.frames.append(image_path)

    def save_gif(self, output_directory=None, plot_name='tetris3d', data_dir='planning/renders'):
        if len(self.frames) < 2:
            # 프레임이 하나 이하이면 GIF 생성 안 함 or 단일 float로 처리
            print("Not enough frames to create a GIF (need at least 2).")
            return

        if output_directory is None:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            output_path = os.path.join(data_dir, plot_name + '.gif')
        else:
            output_path = os.path.join(output_directory, plot_name + '.gif')

        images = [imageio.imread(image) for image in self.frames]

        # 2개 이상 프레임이 있으므로, duration을 리스트로 OK
        durations = [10.0]*len(images)  
        imageio.mimwrite(output_path, images, duration=durations)
        print(f"GIF saved as {output_path}")


    def plot_piece(self, bin_matrix = None, plot_name= None, several_axis_view=False, data_dir = 'planning/renders'):
        '''
        dynamic_display가 True이면 여러 방향
        
        '''
        if bin_matrix is None:
            bin_matrix = self.bin_matrix

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        filled_positions = np.argwhere(bin_matrix != 0)

        for (z, y, x) in filled_positions:
            piece_id = bin_matrix[z, y, x]
            if piece_id == 255:
                color = np.array([169, 169, 169]) / 255.0
            else:
                color = self.tetromino3d.colors[piece_id] / 255.0
            
            ax.bar3d(x, y, z, 1, 1, 1, color=color, alpha=0.15)

        ax.set_xlim(0, bin_matrix.shape[2])
        ax.set_ylim(0, bin_matrix.shape[1])
        ax.set_zlim(0, bin_matrix.shape[0])

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        if plot_name:
            ax.set_title(plot_name)
        
        else:
            ax.set_title("3D Tetris Board")

        if data_dir:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            main_path = os.path.join(data_dir, f"main_{len(self.frames)}_{plot_name}")
            plt.savefig(f"{main_path}.png")
            plt.close(fig)

        plt.show(block=True)

        if several_axis_view is True:
            

            # Save views from X, Y, Z axes
            views = {
                'x': (0, -90),
                'y': (0, 0),
                'z': (90, -90)
            }

            
            for axis, (elev, azim) in views.items():
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                for (z, y, x) in filled_positions:
                    piece_id = bin_matrix[z, y, x]
                    if piece_id == 255:
                        color = np.array([169, 169, 169]) / 255.0
                    else:
                        color = self.tetromino3d.colors[piece_id] / 255.0
                    
                    ax.bar3d(x, y, z, 1, 1, 1, color=color, alpha=0.15)

                ax.set_xlim(0, bin_matrix.shape[2])
                ax.set_ylim(0, bin_matrix.shape[1])
                ax.set_zlim(0, bin_matrix.shape[0])

                ax.view_init(elev=elev, azim=azim)
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                ax.set_title(f"3D Tetris Board - {axis.upper()} View")

                if data_dir:
                    path = os.path.join(data_dir, f"{axis}_{len(self.frames)}")
                    plt.savefig(f"{path}.png")
                plt.close(fig)
        return main_path + '.png'

# if __name__ == '__main__':
#     # width, height, depth = 256, 256, 70
#     width, height, depth = 20,20,20
#     env = Tetris3D(width=width, height=height, depth=depth, render=True, mode='random')
#     env.reset()

#     # while True:
#         # '''
#         # action[0]: 이전아이템의 좌측(0), 우측(1), 뒤(2), 앞(3), 위(4) 위치
#         # action[1]: 0이면 회전안함, 1이면 z축기준 90도 회전
#         # '''
#         # action = [2, 0]
#         # # action = [random.choice(range(width-13)), random.choice(range(height-8)), 0]   # 임의의 x, 임의의 y, 회전 방향
#         # # env.test_bin_matrix()
#         # # _, done = env.step(action)
#         # reward, truncated, terminaled = env.step(action)
#         # done = truncated or terminaled
#         # if done:
#         #     break

#     for i in [(9,9),(1,0), (1,0), (3,0),(3,0),(0,0),(4,0),(4,1), (2,1)]:
#         reward, truncated, terminaled = env.step(i)
#         done = truncated or terminaled
#         if done:
#             env.render()
#             break



        

