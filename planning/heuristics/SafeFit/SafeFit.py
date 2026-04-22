from planning.item import RotationType
from planning.heuristics.base import Base

from utils.item_utils import *
from utils.position import *

class SafeFit(Base):
    '''
    margin이 있는 곳을 채우도록 하여 적재 안정성을 고려한 적재 알고리즘
    '''
    def __init__(self, unfit_stop_setting=True, rotation_type=RotationType.BasicRotation):
        super().__init__(unfit_stop_setting, rotation_type)

    def stack(self, bin, items_list):
        """
        bin에 items_list를 적재하되, composite(부모) 아이템도 생성될 수 있음.
        items_list는 원본 아이템들.
        """

        # 1) 원본 아이템의 ID 기준으로 index_map 생성
        index_map = {it._id: i for i, it in enumerate(items_list)}

        # 2) fit_result 초기화
        fit_result = [-1] * len(items_list)

        # 코너에 보호용 큐브 추가
        if bin.corner != 0 and bin.size == 0:
            corner_list = self.addCorner(bin)
            for i in range(len(corner_list)):
                self.putCorner(bin, i, corner_list[i])

        # 3) Composite 구성
        grouped_items_list = self.group_items_with_top(bin, items_list)
        grouped_items_list.sort(key=lambda x: (not x.is_composite, -(x.width * x.height)))

        # 4) 아이템 적재
        for comp_item in grouped_items_list:
            fitted, _ = self.addItem(bin, comp_item)
            self._mark_fit_result(comp_item, fitted, fit_result, index_map)

        # 5) 실패한 remainder 재도전
        if fit_result.count(1) != len(items_list):
            remainder_items = [items_list[i] for i, res in enumerate(fit_result) if res == 0]
            for item in remainder_items:
                fitted, _ = self.addItem(bin, item)
                if fitted > 0:
                    print(f'{item.name}--------------remainder--------------')
                    self._mark_fit_result(item, fitted, fit_result, index_map)

        if fit_result.count(1) != len(items_list):
            # 그래도 실패 -> surface_ratio 0.7로 하여 호출
            bk_surface_ratio = bin.support_surface_ratio
            bin.support_surface_ratio = 0.7
            remainder_items = [items_list[i] for i, res in enumerate(fit_result) if res == 0]
            if fit_result.count(1) != len(items_list):
                for item in remainder_items:
                    fitted, _ = self.addItem(bin, item)
                    if fitted > 0:
                        print(f'{item.name}--------------bk_surface_ratio1--------------')
                        self._mark_fit_result(item, fitted, fit_result, index_map)
            bin.support_surface_ratio = bk_surface_ratio


        # 실패한 리스트 self.unfit_items에 추가
        for i, res in enumerate(fit_result):
            if res == 0:
                self.unfit_items.add(items_list[i])

        return fit_result

    def calc_max_displacement(self, bin, candidate_item):
        """
        후보 아이템(candidate_item)을 배치했을 때의
        Δ+ₓ, Δ−ₓ, Δ+ᵧ, Δ−ᵧ 를 계산하고
        M = max(Δ+ₓ, Δ−ₓ, Δ+ᵧ, Δ−ᵧ) 를 반환.
        (안정성 지표 1번)
        """
        # 이미 배치된 아이템들
        placed = sorted(bin.get_all_items(), key=lambda it: it._id)

        # A = candidate_item
        xA, yA, _ = candidate_item.b_position
        wA, hA, dA = candidate_item.getDimension()
        mx, my = bin.margin_x, bin.margin_y
        wB, dB = bin.width, bin.depth

        # 벽과의 초기 거리
        Δpx = wB - (xA + wA)
        Δmx = xA
        Δpy = dB - (yA + dA)
        Δmy = yA

        # 다른 아이템들과의 거리 계산
        for B in placed:
            if B._id == candidate_item._id:
                continue
            xB, yB, _ = B.b_position
            wB2, hB2, dB2 = B.getDimension()

            # +X 방향
            if xB >= xA + wA - 1e-9:
                Δpx = min(Δpx, xB - (xA + wA) - mx)
            # -X 방향
            if xB + wB2 <= xA + 1e-9:
                Δmx = min(Δmx, xA - (xB + wB2) - mx)
            # +Y 방향
            if yB >= yA + dA - 1e-9:
                Δpy = min(Δpy, yB - (yA + dA) - my)
            # -Y 방향
            if yB + dB2 <= yA + 1e-9:
                Δmy = min(Δmy, yA - (yB + dB2) - my)

        # 음수(겹침)는 0으로 클램핑
        Δpx, Δmx, Δpy, Δmy = map(lambda v: max(v, 0), (Δpx, Δmx, Δpy, Δmy))

        # M(p) = 최대 이동거리
        vec = round(max(Δpx, Δmx, Δpy, Δmy), 4)
        candidate_item.options['vec'] = vec
        return vec

    def addItem(self, bin, item, test=False):
        self.set_init_PT(bin, bin.get_all_items())


        # 2) 배치 가능한 pivot 후보들 수집
        feasible_pivots = []
        i=0
        for pivot in bin.pivotTree.in_order_traversal():
            i+=1
            fitted, loaded_item = checkPivot_R(bin, item, [pivot.x, pivot.y, pivot.z], pivot.rt)
            if fitted > 0:
                feasible_pivots.append((pivot, loaded_item))
        # print(i, bin.pivotTree.size)
        if not feasible_pivots:
            # 후보가 전혀 없다면 실패
            return False, None
        
        # Δ 계산 (모든 아이템 대상)
        def calc_delta_vector(bin, candidate_item):
            """
            bin + candidate_item 배치 결과에 대한 Δ 벡터 반환
            벡터 순서는 item_id(또는 적재 순) 기준 고정해야 함.
            """
            placed = sorted(bin.get_all_items(), key=lambda it: it._id)
            full_list = placed + [candidate_item]

            vec = []
            mx, my = bin.margin_x, bin.margin_y
            wB, dB = bin.width, bin.depth

            tol = 1e-9

            for A in full_list:
                xA, yA, zA = A.b_position
                wA, hA, dA = A.getDimension()  # w: X, h: Z, d: Y

                # 벽과의 초기 거리
                Δpx = wB - (xA + wA)   # +X
                Δmx = xA               # -X
                Δpy = dB - (yA + dA)   # +Y
                Δmy = yA               # -Y

                # 다른 아이템과의 거리 업데이트
                for B in placed:
                    if B._id == A._id:
                        continue

                    xB, yB, zB = B.b_position
                    wB2, hB2, dB2 = B.getDimension()

                    # 구간 겹침 검사 헬퍼
                    def overlap(a0, a1, b0, b1):
                        return (a0 < b1 - tol) and (b0 < a1 - tol)

                    # +X 방향: A 오른쪽에 있는 B 중, Y·Z 오버랩 시
                    if xB >= xA + wA - tol \
                    and overlap(yA, yA + dA, yB, yB + dB2) \
                    and overlap(zA, zA + hA, zB, zB + hB2):
                        Δpx = min(Δpx, xB - (xA + wA) - mx)

                    # -X 방향: A 왼쪽에 있는 B 중, Y·Z 오버랩 시
                    if xB + wB2 <= xA + tol \
                    and overlap(yA, yA + dA, yB, yB + dB2) \
                    and overlap(zA, zA + hA, zB, zB + hB2):
                        Δmx = min(Δmx, xA - (xB + wB2) - mx)
                    
                    Δpx = min(Δpx, Δmx)

                    # +Y 방향: A 앞에 있는 B 중, X·Z 오버랩 시
                    if yB >= yA + dA - tol \
                    and overlap(xA, xA + wA, xB, xB + wB2) \
                    and overlap(zA, zA + hA, zB, zB + hB2):
                        Δpy = min(Δpy, yB - (yA + dA) - my)

                    # -Y 방향: A 뒤에 있는 B 중, X·Z 오버랩 시
                    if yB + dB2 <= yA + tol \
                    and overlap(xA, xA + wA, xB, xB + wB2) \
                    and overlap(zA, zA + hA, zB, zB + hB2):
                        Δmy = min(Δmy, yA - (yB + dB2) - my)
                    
                    Δpy = min(Δpy, Δmy)

                # 음수(겹침)인 경우 0으로 클램핑
                # Δpx, Δmx, Δpy, Δmy = map(lambda v: max(v, 0), (Δpx, Δmx, Δpy, Δmy))
                Δpx, Δpy = map(lambda v: max(v, 0), (Δpx, Δpy))


                # Δ-벡터는 [Δ+X, Δ+Y, Δ−X, Δ−Y, …] 순서로 full_list 길이만큼 쌓는다
                vec.extend([Δpx, Δpy])
                candidate_item.options['vec'] = vec

            # noise 제거 및 튜플로 반환
            return tuple(np.round(vec, 4))

        def pivot_sort_key(tup):
            pivot, loaded_item = tup

            # ① Δ-vector (tuple)
            delta_vec = calc_delta_vector(bin, loaded_item)
            # ② pivot x, ③ pivot y (오름차순)
            px = pivot.x
            py = pivot.y

            return (pivot.z, delta_vec)

        feasible_pivots.sort(key=pivot_sort_key)
        # feasible_pivots.sort(
        #     key=lambda tup: (
        #         # 1) M(p)
        #         self.calc_max_displacement(bin, tup[1]),
        #         # (옵션) tie-breaker로 pivot.z, pivot.x, pivot.y 사용
        #         tup[0].z, tup[0].x, tup[0].y
        #     )
        # )
        for i, (pivot, loaded_item) in enumerate(feasible_pivots):
            # 1) pivot.z (오름차순)
            # 2) Δ-vector (tuple)
            # 3) pivot.x (오름차순)
            # 4) pivot.y (오름차순)
            print(f'{i}: {pivot.z}, {loaded_item.options["vec"]}, {pivot.x}, {pivot.y}')

        # 정렬 결과 0번째가 "가장 좋은" pivot
        best_pivot, best_loaded_item = feasible_pivots[0]

        # 4) bin에 최종 저장 + pivot 관련 처리
        bin.store(best_loaded_item)
        # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
        best_loaded_item.update_child_positions_z()
        self.store2Pivot(bin)
        # bin.render(name = 'pivot', save=True, show=True)
        return True, best_loaded_item