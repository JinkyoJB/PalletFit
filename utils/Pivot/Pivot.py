import numpy as np

class Pivot():
    """
    아이템 배치 가치판단 객체 (동적 지표는 필요할 때마다 계산)
    """
    def __init__(self, x, y, z, rt, direction="any", bench_bin=None, options=None):
        '''
        연산에 기준이 되는 bin = bench_bin
        '''
        self.x =  np.float64(x)
        self.y = np.float64(y)
        self.z = np.float64(z)
        self.rt = [np.float64(v) for v in rt]  # 쿼터니언 회전 벡터
        self.direction = direction
        self.bench_bin = bench_bin  # 벤치마크용 bin (필요시 사용)
        self.options = options if options is not None else {}
        # 현재 pivot에서 놓을 수 있는 최대크기의 아이템 [w, h, d]

    def get_max_item_size(self):
        """
        현재 pivot에서 놓을 수 있는 최대크기의 아이템 [w, h, d]를 반환
        """
        self.max_item_size = self.bench_bin.get_max_item_size(self.x, self.y, self.z) if self.bench_bin else None


 