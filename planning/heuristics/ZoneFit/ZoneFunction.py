from planning.heuristics.ZoneFit.Zone import Zone


def get_init_zone(bin):
    """
    bin을 하나의 zone으로 초기화
    """
    return Zone(0,0,0, bin.width, bin.height, bin.depth, 0, max_weight=bin.max_weight, name='init_zone')


def get_adjacent_zones(zone_list, zone):
    """
    zone_list에서 zone과 인접한 zone들을 반환.
    사선 접촉은 제외하며, 면을 공유하는 경우만 포함.
    """
    adjacent_zones = []
    zone_vertices = set(tuple(v) for v in zone.getVertices())  # 현재 zone의 꼭짓점을 집합으로 변환
    for sz in zone_list:
        if sz is zone:
            continue
        sz_vertices = set(tuple(v) for v in sz.getVertices())  # 비교할 zone의 꼭짓점을 집합으로 변환
        shared_vertices = zone_vertices & sz_vertices  # 두 zone의 공유된 꼭짓점 찾기

        if len(shared_vertices) < 4:
            continue  # 공유된 꼭짓점이 4개 미만이면 제외
        adjacent_zones.append(sz)

    return adjacent_zones


def addItem2Zone(zone, item):
    """
    아이템과 zone의 교집합을 계산하여 배치하는 함수.
      - 완전히 들어가면 -> zone.store(item)
      - 일부 겹치면 -> child item 생성 및 store
      - 전혀 안 겹치면 -> 아무 동작 안 함
    """

    # Zone의 3D 범위
    Zx_min, Zy_min, Zz_min = zone.x, zone.y, zone.z
    Zw, Zh, Zd = zone.width, zone.height, zone.depth
    Zx_max, Zy_max = Zx_min + Zw, Zy_min + Zh
    # z축 범위 [0, Zd] 가정

    # Item의 3D 범위
    ix, iy, iz = item.b_position
    iw, ih, id_ = item.getDimension()
    x_min, x_max = ix, ix + iw
    y_min, y_max = iy, iy + ih
    z_min, z_max = iz, iz + id_

    # 교집합 계산
    rx_min = max(Zx_min, x_min)
    rx_max = min(Zx_max, x_max)
    ry_min = max(Zy_min, y_min)
    ry_max = min(Zy_max, y_max)
    rz_min = max(0,      z_min)
    rz_max = min(Zd,     z_max)

    # 유효 교집합 여부
    if (rx_min < rx_max) and (ry_min < ry_max) and (rz_min < rz_max):
        # 부분 or 완전 겹침
        fully_contained = (
            rx_min == x_min and rx_max == x_max and
            ry_min == y_min and ry_max == y_max and
            rz_min == z_min and rz_max == z_max
        )
        if fully_contained:
            # 완전히 들어감
            zone.store(item)

def split_zone(zone, split_x, split_y, split_z):
    """
    zone 내부를 'subzone의 크기' 관점으로 8개 구역으로 나누어 반환한다.
    
    즉, split_x, split_y, split_z가 각각 subzone의 (width, height, depth) 역할을 하며,
    나머지 구역은 (원본 크기 - split_x, etc)로 분할.
    
    만약 split_x > zone.width 등으로 인해 (w - split_x)가 음수가 되면, 해당 서브존은 무효가 된다.
    """
    x1, y1, z1 = zone.x, zone.y, zone.z
    w,  h,  d  = zone.width, zone.height, zone.depth

    # subzone1, 아이템 영역
    subzone1 = Zone(
        x1,
        y1,
        z1,
        split_x,
        split_y,
        split_z,
        name=f"sz1_{zone.x}_{zone.y}_{zone.z}"
    )

    # subzone2, 아이템의 오른쪽 영역
    subzone2 = Zone(
        x1 + split_x,
        y1,
        z1,
        w - split_x,
        split_y,
        split_z,
        name=f"sz2_{zone.x}_{zone.y}_{zone.z}"
    )

    # subzone3, 아이템의 뒤쪽 영역
    subzone3 = Zone(
        x1,
        y1 + split_y,
        z1,
        split_x,
        h - split_y,
        split_z,
        name=f"sz3_{zone.x}_{zone.y}_{zone.z}"
    )

    # subzone4, 아이템의 대각선 영역
    subzone4 = Zone(
        x1 + split_x,
        y1 + split_y,
        z1,
        w - split_x,
        h - split_y,
        split_z,
        name=f"sz4_{zone.x}_{zone.y}_{zone.z}"
    )

    # subzone5, 아이템의 위쪽 영역
    subzone5 = Zone(
        x1,
        y1,
        z1 + split_z,
        split_x,
        split_y,
        d - split_z,
        name=f"sz5_{zone.x}_{zone.y}_{zone.z}"
    )

    # subzone6, 아이템의 위쪽 오른쪽 영역
    subzone6 = Zone(
        x1 + split_x,
        y1,
        z1 + split_z,
        w - split_x,
        split_y,
        d - split_z,
        name=f"sz6_{zone.x}_{zone.y}_{zone.z}"
    )

    # subzone7, 아이템의 위쪽 뒤쪽 영역
    subzone7 = Zone(
        x1,
        y1 + split_y,
        z1 + split_z,
        split_x,
        h - split_y,
        d - split_z,
        name=f"sz7_{zone.x}_{zone.y}_{zone.z}"
    )

    # subzone8, 아이템의 위쪽 대각선 영역
    subzone8 = Zone(
        x1 + split_x,
        y1 + split_y,
        z1 + split_z,
        w - split_x,
        h - split_y,
        d - split_z,
        name=f"sz8_{zone.x}_{zone.y}_{zone.z}"
    )

    zones = [
        subzone1, subzone2, subzone3, subzone4,
        subzone5, subzone6, subzone7, subzone8
    ]

    # (옵션) 음수 또는 0 크기의 subzone을 제외
    valid_zones = []
    for z_ in zones:
        if z_.width > 0 and z_.height > 0 and z_.depth > 0:
            valid_zones.append(z_)

    return valid_zones


def merge_shared_face_zones(tree):
    """
    1단계: z축 (윗면-아랫면) 끼리 우선 병합
    2단계: 그 외 면(앞/뒤/왼/오른) 병합
    더 이상 병합할 게 없을 때까지 반복.
    """
    def zones_share_top_bottom(zone1, zone2):
        """
        zone1의 윗면 == zone2의 아랫면 (또는 반대)인지 확인.
        그리고 x, y 범위가 정확히 겹치는지(=면 공유) 판별.

        zone1 윗면 z = zone1.z + zone1.depth
        zone2 아랫면 z = zone2.z
        => 같으면 윗면=아랫면
        => x, y 범위도 완전히 같아야 '면' 공유로 간주
        """
        z1_top = zone1.z + zone1.depth
        z1_bottom = zone1.z
        z2_top = zone2.z + zone2.depth
        z2_bottom = zone2.z

        # case 1) zone1 top == zone2 bottom
        #   => x, y 범위 (x~x+width, y~y+height)가 동일해야 함
        if abs(z1_top - z2_bottom) < 1e-12:
            # x range, y range 비교
            if (abs(zone1.x - zone2.x) < 1e-12 and
                abs((zone1.x + zone1.width) - (zone2.x + zone2.width)) < 1e-12 and
                abs(zone1.y - zone2.y) < 1e-12 and
                abs((zone1.y + zone1.height) - (zone2.y + zone2.height)) < 1e-12):
                return True

        # case 2) zone2 top == zone1 bottom
        if abs(z2_top - z1_bottom) < 1e-12:
            if (abs(zone1.x - zone2.x) < 1e-12 and
                abs((zone1.x + zone1.width) - (zone2.x + zone2.width)) < 1e-12 and
                abs(zone1.y - zone2.y) < 1e-12 and
                abs((zone1.y + zone1.height) - (zone2.y + zone2.height)) < 1e-12):
                return True

        return False

    def zones_share_side_face(zone1, zone2):
        """
        기존 로직: 꼭짓점 4개 공유 -> 한 면 공유
        (x축이나 y축 방향 면)
        """
        vertices1 = set(tuple(v) for v in zone1.getVertices())
        vertices2 = set(tuple(v) for v in zone2.getVertices())
        shared_vertices = vertices1 & vertices2
        return (len(shared_vertices) == 4)

    def merge_top_bottom_pass():
        """
        전체 zone 중 '윗면-아랫면'을 공유하는 pairs를 하나 찾아 병합.
        하나라도 병합하면 True, 없으면 False
        """
        zones = tree.get_sorted_zones_leftover()
        for i in range(len(zones)):
            z1 = zones[i]
            # z1과 인접한 후보 찾기
            adj = get_adjacent_zones(zones, z1)
            for z2 in adj:
                if zones_share_top_bottom(z1, z2):
                    tree.merge_zones(z1, z2)
                    # print(f"[merge_top_bottom_pass] merged top/bottom of '{z1.name}' & '{z2.name}'")
                    return True  # 한 번 병합 후 종료
        return False

    def merge_side_face_pass():
        """
        전체 zone 중 '옆면(앞/뒤/왼/오)'을 공유하는 pairs를 하나 찾아 병합.
        하나라도 병합하면 True, 없으면 False
        """
        zones = tree.get_sorted_zones_leftover_desc()
        for i in range(len(zones)):
            z1 = zones[i]
            adj = get_adjacent_zones(zones, z1)
            adj.sort(key=lambda x: (x.leftover), reverse=True)
            for z2 in adj:
                if zones_share_side_face(z1, z2):
                    tree.merge_zones(z1, z2)
                    # print(f"[merge_side_face_pass] merged side face of '{z1.name}' & '{z2.name}'")
                    return True
        return False

    while True:
        # 1단계: top/bottom 병합 우선 시도
        merged_tb = merge_top_bottom_pass()
        if merged_tb:
            continue  # 또다른 top/bottom 병합이 있을 수 있으므로 다시 시도

        # 2단계: side-face 병합 시도
        merged_side = merge_side_face_pass()
        if merged_side:
            continue  # 병합 후 다시 전체 재검사

        # 둘 다 못 하면 종료
        break

