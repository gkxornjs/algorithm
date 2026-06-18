# ============================================================
# main.py
# 재난 대피 경로 안내 시스템 - 통합 실행 및 Streamlit UI 모듈
#
# 실행 환경: VSCode / Python 3.9+
#   - 통합 테스트: python main.py
#   - UI 실행:     streamlit run main.py
#
# 필요 라이브러리: requirements.txt 참고
#   matplotlib, numpy, Pillow, streamlit, streamlit-image-coordinates
#
# 자료구조: 2D 배열(맵/화재/연기), 딕셔너리(session_state), 집합(set), 우선순위 큐(A* 내부)
# 알고리즘: A*(최단 경로), 다중 층 A*(층간 이동), BFS(화재/연기 확산),
#           유니온-파인드(연결성 확인), 퀵 정렬(탈출 시간), 이진 탐색(목표 시간 내 인원)
#
# Input 데이터: building_map.txt — 직접 구성 (가천대학교 AI공학관 기반)
# ============================================================

import os
import copy
import random
import functools
import matplotlib
matplotlib.use('Agg')   # GUI 없는 환경에서 plt.show() 블로킹 방지
import numpy as np
from PIL import Image
from map_builder import build_map
from fire_spread import (
    simulate_fire,
    simulate_fire_spread,
    get_random_fire_positions,
    get_exit_fire_times,
)
from path_finder import get_escape_path, find_best_exit, find_multi_floor_path
from evacuation import (
    create_evacuee_dict,
    update_evacuee_result,
    print_evacuation_statistics,
    quick_sort_escape_times,
    binary_search_time,
    get_evacuation_statistics,
)
from risk_analyzer import (
    calculate_position_risk,
    calculate_path_risk,
    get_risk_level,
)
from smoke_spread import (
    simulate_smoke_spread,
    get_smoke_cells_at_time,
)

MAP_FILE = os.path.join(os.path.dirname(__file__), "building_map.txt")

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAP_FILES = {
    3: os.path.join(_DATA_DIR, "floor3_map.txt"),
    4: os.path.join(_DATA_DIR, "floor4_map.txt"),
    5: os.path.join(_DATA_DIR, "floor5_map.txt"),
}

PASSABLE = {'.', 'R', 'X', 'E', 'S', 'F', 'T'}

# 계단 셀 위치 (모든 층 동일)
STAIR_POSITIONS = [(5, 5), (50, 5)]

# ── 다중 층 3D 빌딩 상수 ──────────────────────────────────────────────────────
# 층간 3.2 / 벽은 키 크고 반투명(0.28) → 화재·연기가 측면 투시로 보임
FLOOR_BASES = {3: 0.0, 4: 3.2, 5: 6.4}     # 층별 Z 기준
WALL_H  = 2.80    # 벽 높이 (방 느낌이 나도록 충분히 높음)
FLOOR_T = 0.12    # 바닥 슬래브 두께
STAIR_H = 3.20    # 계단 높이 (정확히 윗 층 바닥)

# 셀 타입별 색상
AC = dict(
    wall  = '#8899aa',   # 청회색 벽 (반투명 처리)
    floor = '#f5f0e8',   # 밝은 아이보리 복도
    room  = '#fffdf8',   # 밝은 흰 강의실
    stair = '#cc7700',   # 호박색 계단
    exit  = '#00cc55',   # 녹색 비상구
    elev  = '#2288ee',   # 파란 엘리베이터
    start = '#66ff22',   # 밝은 연두 시작
    fire  = '#ff2200',   # 빨강 화재
    smoke = '#667799',   # 청회색 연기
    path  = '#ffcc00',   # 금색 경로
    trail = '#dd55ff',   # 보라 자취
    prev  = '#ff7722',   # 주황 화재예정지
)


def pick_random_start(grid):
    candidates = [
        (r, c)
        for r, row in enumerate(grid)
        for c, cell in enumerate(row)
        if cell in ('.', 'R')
    ]
    return random.choice(candidates)


def set_start(grid, pos):
    r, c = pos
    grid[r][c] = 'S'


def run_integration_test():
    print("=" * 60)
    print(" 재난 대피 경로 안내 시스템 — 통합 테스트")
    print("=" * 60)

    # Step 1: 맵 로드
    grid, graph, exits = build_map(MAP_FILE)
    print(f"[1] 맵 로드 완료: {len(grid)}행 × {len(grid[0])}열")
    print(f"    비상구(X): {exits}")

    # Step 2: 시작 위치 설정
    start = pick_random_start(grid)
    set_start(grid, start)
    _, graph, exits = build_map(MAP_FILE, start=start)
    print(f"[2] 시작 위치: {start}")

    # Step 3: 화재 시뮬레이션 (화재 지점 2개)
    fire_time = simulate_fire(grid, fire_count=2)
    print(f"[3] 화재 확산 시뮬레이션 완료 (화재 지점 2곳)")
    get_exit_fire_times(fire_time, exits)

    # Step 4: 대피자 등록 (테스트용 3명)
    evacuee_list = [
        (1, "김민재", start),
        (2, "이태권", start),
        (3, "박지원", start),
    ]
    evacuees = create_evacuee_dict(evacuee_list)
    print(f"\n[4] 대피자 {len(evacuees)}명 등록 완료")

    # Step 5: 각 시간 t=0, 3, 6 에서 경로 탐색
    for t in [0, 3, 6]:
        print(f"\n{'─'*50}")
        print(f" t={t} 시점 탈출 경로 탐색")
        print(f"{'─'*50}")

        path, dist, possible = get_escape_path(grid, graph, fire_time, current_time=t)

        if possible and path:
            print(f"탈출 가능: 경로 길이 {dist}칸")
            print(f"경로 (처음 5칸): {path[:5]} ...")

            for eid in evacuees:
                evacuees = update_evacuee_result(evacuees, eid, path, fire_time)
        else:
            print("탈출 불가 — 모든 출구가 화재로 차단")

    # Step 6: 통계 출력
    print()
    print_evacuation_statistics(evacuees, target_time=20)

    print("\n[완료] 모든 모듈 연동 성공")


# ============================================================
# Streamlit UI (담당: 지원)
# 자료구조/알고리즘 모듈 시각화 — streamlit run main.py
# ============================================================

INF = float('inf')
PASSABLE_START = ('.', 'R')   # 사용자 시작 가능 셀

# README 색상 기준 (path_finder.visualize 색 체계 재사용)
C_WALL    = [0.20, 0.20, 0.20]   # 벽 (#)
C_FIRE    = [0.90, 0.30, 0.20]   # 화재 셀
C_START   = [0.20, 0.70, 0.40]   # 시작 (S)
C_EXIT    = [0.95, 0.75, 0.15]   # 비상구 (X)
C_ELEV    = [0.20, 0.50, 0.90]   # 엘리베이터 (E)
C_PATH    = [1.00, 0.60, 0.10]   # 최단 경로 (*)
C_VISITED = [0.75, 0.85, 1.00]   # A* 탐색 노드
C_FLOOR   = [0.96, 0.96, 0.96]   # 일반 통로
C_CHAR    = [0.55, 0.10, 0.80]   # 대피자(캐릭터) 현재 위치
C_TRAIL   = [0.80, 0.70, 0.92]   # 지나온 자취
C_FIREPT  = [0.65, 0.10, 0.05]   # 수동 지정 화재 발화점(시작 전 미리보기)
C_SMOKE   = [0.60, 0.60, 0.60]   # 연기
C_STAIR   = [1.00, 0.65, 0.00]   # 계단 (T)

FIRE_OK = ('.', 'R')   # 수동 화재 지정 가능 셀


def load_base_map(floor=5):
    """맵 1회 로드 — grid_base(원본), exits 반환."""
    fname = MAP_FILES.get(floor, MAP_FILES[5])
    grid_base, _graph, exits = build_map(fname)
    return grid_base, exits


@functools.lru_cache(maxsize=None)
def _cached_floor_grid(fname):
    """파일명 기준으로 grid를 캐싱 (immutable tuple 반환)."""
    grid, _, _ = build_map(fname)
    return tuple(tuple(row) for row in grid)


def load_all_floor_grids():
    """3개 층 grid를 딕셔너리로 반환. lru_cache로 I/O 최소화."""
    return {f: [list(r) for r in _cached_floor_grid(p)] for f, p in MAP_FILES.items()}


def make_grid_with_start(grid_base, start):
    """원본을 보존하기 위해 깊은 복사 후 시작 위치 'S' 표기 (렌더 마커용)."""
    grid = copy.deepcopy(grid_base)
    r, c = start
    if grid[r][c] in PASSABLE_START:
        grid[r][c] = 'S'
    return grid


def generate_fire(grid_base, fire_count):
    """랜덤 화재 발생 후 BFS 확산 시뮬레이션. fire_time/positions/log 반환.
    알고리즘: BFS (화재 확산) — fire_spread.simulate_fire_spread 호출
    """
    positions = get_random_fire_positions(grid_base, fire_count)
    fire_time, fire_log = simulate_fire_spread(grid_base, positions)
    return fire_time, positions, fire_log


def build_color_array(grid, fire_time, path, start, exits, current_time,
                      char_pos=None, trail=None, fire_preview=None,
                      trail_on_top=False, smoke_time=None):
    """
    맵을 rows×cols×3 RGB(float 0~1) 배열로 변환.
    화재 도달은 현재 시각(current_time) 기준. 대피자(char_pos)·지나온 자취(trail)·
    예정 경로(path)·수동 화재 발화점(fire_preview)을 함께 그린다.
    trail_on_top=True면(탈출 완료 후) 자취를 화재 위에 그려 가려진 경로까지 보여준다.

    자료구조: 2D 배열(fire_time, smoke_time), 집합(set) — 경로/자취 빠른 조회
    """
    rows, cols = len(grid), len(grid[0])
    # 자료구조: 집합(set) — O(1) 조회를 위한 경로·자취·미리보기 집합화
    path_set    = set(path) if path else set()
    trail_set   = set(trail) if trail else set()
    preview_set = set(fire_preview) if fire_preview else set()

    # 자료구조: 2D 배열(img) — rows×cols×3 RGB 픽셀 배열 (numpy)
    img = np.zeros((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            pos  = (r, c)
            if cell == '#':
                img[r][c] = C_WALL
            elif char_pos is not None and pos == tuple(char_pos):
                img[r][c] = C_CHAR            # 대피자 현재 위치 (최우선)
            elif trail_on_top and pos in trail_set:
                img[r][c] = C_TRAIL           # 탈출 완료 후: 화재보다 우선해 경로 표시
            elif fire_time[r][c] != -1 and fire_time[r][c] <= current_time:
                img[r][c] = C_FIRE            # 현재 시각 기준 화재 도달 (-1=미도달 제외)
            elif smoke_time is not None and smoke_time[r][c] != -1 and smoke_time[r][c] <= current_time:
                img[r][c] = C_SMOKE
            elif pos in preview_set:
                img[r][c] = C_FIREPT          # 시작 전 수동 화재 발화점 미리보기
            elif cell == 'X':
                img[r][c] = C_EXIT
            elif pos in path_set:
                img[r][c] = C_PATH            # 지금 위치에서 출구까지 예정 경로
            elif pos in trail_set:
                img[r][c] = C_TRAIL           # 지나온 자취
            elif cell == 'S' or pos == tuple(start):
                img[r][c] = C_START
            elif cell == 'E':
                img[r][c] = C_ELEV
            elif cell == 'T':
                img[r][c] = C_STAIR
            else:
                img[r][c] = C_FLOOR
    return img


def build_map_image(grid, fire_time, path, start, exits, current_time, cell_px=12,
                    char_pos=None, trail=None, fire_preview=None, trail_on_top=False, smoke_time=None):
    """
    색상 배열을 셀당 cell_px 픽셀로 확대한 PIL 이미지로 변환.
    클릭 좌표 → 셀 매핑이 선형이 되도록 축/여백 없이 순수 픽셀로 렌더한다.
    """
    img = build_color_array(grid, fire_time, path, start, exits, current_time,
                            char_pos=char_pos, trail=trail, fire_preview=fire_preview,
                            trail_on_top=trail_on_top, smoke_time=smoke_time)
    arr = (img * 255).astype(np.uint8)
    big = np.repeat(np.repeat(arr, cell_px, axis=0), cell_px, axis=1)
    return Image.fromarray(big)


def _swatch(color, label):
    rgb = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
    return (f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{rgb};border:1px solid #999;"
            f"margin:0 4px -1px 8px'></span>{label}")


def legend_html():
    """클릭형 이미지에는 범례가 없으므로 HTML 색상 범례를 따로 그린다."""
    items = [
        (C_CHAR, '대피자'), (C_START, '시작(S)'), (C_EXIT, '비상구(X)'),
        (C_FIRE, '화재'), (C_PATH, '예정 경로'), (C_TRAIL, '지나온 길'),
        (C_ELEV, '엘리베이터(E)'), (C_STAIR, '계단(T)'), (C_WALL, '벽(#)'), (C_SMOKE, '연기'),
    ]
    return "".join(_swatch(c, l) for c, l in items)


def build_heatmap_image(grid, fire_time, evacuee_paths=None, cell_px=12):
    """
    위험도 히트맵: 화재 도달 속도(65%) + 경로 병목 빈도(35%) 결합.
    score = danger * 0.65 + bottleneck * 0.35
    빨강=위험 / 초록=안전 / 파랑=비상구 / 주황=계단

    자료구조: 2D 배열(fire_time, freq), numpy 배열(img)
    알고리즘: 위험도 점수 계산 (정규화 + 가중 합산)
    """
    rows, cols = len(grid), len(grid[0])

    valid_ft = [
        fire_time[r][c]
        for r in range(rows) for c in range(cols)
        if fire_time[r][c] not in (-1, float('inf')) and fire_time[r][c] > 0
    ]
    max_ft = max(valid_ft) if valid_ft else 1.0

    freq = [[0] * cols for _ in range(rows)]
    if evacuee_paths:
        for path in evacuee_paths:
            for pos in path:
                r, c = pos
                if 0 <= r < rows and 0 <= c < cols:
                    freq[r][c] += 1
    max_freq = max(freq[r][c] for r in range(rows) for c in range(cols)) or 1

    img = np.zeros((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell == '#':
                img[r][c] = [0.15, 0.15, 0.15]
                continue
            if cell == 'X':
                img[r][c] = [0.0, 0.4, 1.0]
                continue
            if cell == 'T':
                img[r][c] = [1.0, 0.65, 0.0]
                continue

            ft = fire_time[r][c]
            danger = 0.0 if ft in (-1, float('inf')) else 1.0 - min(ft / max_ft, 1.0)
            bottle = freq[r][c] / max_freq if evacuee_paths else 0.0
            score  = min(danger * 0.65 + bottle * 0.35, 1.0)

            if score < 0.5:
                t2 = score * 2
                img[r][c] = [t2, 1.0, 0.0]
            else:
                t2 = (score - 0.5) * 2
                img[r][c] = [1.0, 1.0 - t2, 0.0]

    arr = (img * 255).astype(np.uint8)
    big = np.repeat(np.repeat(arr, cell_px, axis=0), cell_px, axis=1)
    return Image.fromarray(big)


def _make_box_mesh(positions, z_bottom=0.0, z_top=1.0, cell_frac=1.0):
    """셀 위치 목록을 단일 Mesh3d 박스 메시 데이터로 일괄 변환.
    cell_frac < 1.0 이면 셀 중앙에 작은 기둥으로 렌더링."""
    vx, vy, vz = [], [], []
    ti, tj, tk = [], [], []
    pad = (1.0 - cell_frac) / 2.0
    for idx, (r, c) in enumerate(positions):
        b = idx * 8
        x0, x1 = c + pad, c + 1 - pad
        y0, y1 = r + pad, r + 1 - pad
        vx += [x0, x1, x1, x0, x0, x1, x1, x0]
        vy += [y0, y0, y1, y1, y0, y0, y1, y1]
        vz += [z_bottom] * 4 + [z_top] * 4
        for fi, fj, fk in [
            (0,1,2),(0,2,3),
            (4,5,6),(4,6,7),
            (0,1,5),(0,5,4),
            (1,2,6),(1,6,5),
            (2,3,7),(2,7,6),
            (0,3,7),(0,7,4),
        ]:
            ti.append(b + fi); tj.append(b + fj); tk.append(b + fk)
    return vx, vy, vz, ti, tj, tk


def build_3d_map_figure(grid, fire_time, path, start, exits, current_time,
                         char_pos=None, trail=None, fire_preview=None, smoke_time=None):
    """
    맵을 Plotly 3D Mesh로 렌더링.
    벽은 높은 박스, 화재/경로/연기 등은 높이별 컬러 타일로 표현.
    """
    import plotly.graph_objects as go

    rows, cols = len(grid), len(grid[0])
    path_set    = set(map(tuple, path))         if path         else set()
    trail_set   = set(map(tuple, trail))        if trail        else set()
    preview_set = set(map(tuple, fire_preview)) if fire_preview else set()
    exit_set    = {tuple(e) for e in exits}

    buckets = {k: [] for k in
               ('wall', 'fire', 'smoke', 'path', 'trail',
                'exit', 'start', 'elev', 'stair', 'floor', 'room', 'preview')}

    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            pos  = (r, c)
            if cell == '#':
                buckets['wall'].append(pos)
            elif char_pos is not None and pos == tuple(char_pos):
                pass  # Scatter3d로 별도 처리
            elif fire_time[r][c] != -1 and fire_time[r][c] <= current_time:
                buckets['fire'].append(pos)   # -1=미도달 제외
            elif (smoke_time is not None
                  and smoke_time[r][c] != -1
                  and smoke_time[r][c] <= current_time):
                buckets['smoke'].append(pos)
            elif pos in preview_set:
                buckets['preview'].append(pos)
            elif pos in path_set:
                buckets['path'].append(pos)
            elif pos in trail_set:
                buckets['trail'].append(pos)
            elif pos in exit_set or cell == 'X':
                buckets['exit'].append(pos)
            elif cell == 'S' or pos == tuple(start):
                buckets['start'].append(pos)
            elif cell == 'E':
                buckets['elev'].append(pos)
            elif cell == 'T':
                buckets['stair'].append(pos)
            elif cell == 'R':
                buckets['room'].append(pos)
            else:
                buckets['floor'].append(pos)

    # (버킷키, z높이, 색상, 불투명도, 범례명)
    # 벽을 낮게(0.40) 유지해 위에서 내려다볼 때 화재·연기가 잘 보이도록
    SPEC = [
        ('floor',   0.02, '#f5e6b0', 1.00, '복도'       ),
        ('room',    0.04, '#a8c8f0', 1.00, '강의실'      ),
        ('trail',   0.06, '#dd66ff', 1.00, '지나온 길'   ),
        ('path',    0.10, '#ffdd00', 1.00, '예정 경로'   ),
        ('exit',    0.12, '#00ee66', 1.00, '비상구(X)'   ),
        ('stair',   0.14, '#ff9900', 1.00, '계단(T)'     ),
        ('start',   0.14, '#88ff22', 1.00, '시작(S)'     ),
        ('elev',    0.14, '#00ccff', 1.00, '엘리베이터'  ),
        ('preview', 0.18, '#ff7722', 1.00, '화재 예정지' ),
        ('smoke',   0.32, '#556677', 0.82, '연기'        ),  # 진한 회색, 불투명하게
        ('fire',    0.46, '#ff2200', 1.00, '화재'        ),
        ('wall',    0.50, '#c8c8c8', 1.00, '벽'          ),  # 낮은 벽 — 위에서 보기 용이
    ]

    # ambient 높여서 전체적으로 밝게, 측면도 잘 보이도록
    mc_lighting = dict(ambient=0.65, diffuse=0.85, roughness=0.4, specular=0.15, fresnel=0.05)
    mc_light_pos = dict(x=100, y=-300, z=800)

    fig = go.Figure()

    for key, z_top, color, opacity, name in SPEC:
        positions = buckets[key]
        if not positions:
            continue
        vx, vy, vz, t_i, t_j, t_k = _make_box_mesh(positions, z_top=z_top)
        fig.add_trace(go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=t_i, j=t_j, k=t_k,
            color=color, opacity=opacity,
            name=name, showlegend=True,
            flatshading=True,
            lighting=mc_lighting,
            lightposition=mc_light_pos,
        ))

    # 예정 경로 선 (금빛 라인)
    if path and len(path) >= 2:
        fig.add_trace(go.Scatter3d(
            x=[p[1] + 0.5 for p in path],
            y=[p[0] + 0.5 for p in path],
            z=[0.14] * len(path),
            mode='lines',
            line=dict(color='#ffcc00', width=7),
            name='경로 선', showlegend=False,
        ))

    # 대피자 — 밝은 보라 다이아몬드
    if char_pos is not None:
        cr, cc = char_pos
        fig.add_trace(go.Scatter3d(
            x=[cc + 0.5], y=[cr + 0.5], z=[0.70],
            mode='markers',
            marker=dict(size=12, color='#cc44ff', symbol='diamond',
                        line=dict(color='white', width=2)),
            name='대피자', showlegend=True,
        ))

    fig.update_layout(
        scene=dict(
            uirevision='3d_map',   # 재렌더링 시 카메라 시점 유지
            xaxis=dict(visible=False, range=[0, cols]),
            yaxis=dict(visible=False, range=[rows, 0]),
            zaxis=dict(visible=False, range=[0, 0.75]),
            aspectmode='manual',
            aspectratio=dict(
                x=cols / max(rows, cols),
                y=rows / max(rows, cols),
                z=0.10,   # 벽이 낮아졌으므로 z 비율도 축소
            ),
            camera=dict(
                eye=dict(x=0.0, y=-0.2, z=2.2),   # 기본: 거의 수직 내려다보기
                up=dict(x=0, y=0, z=1),
            ),
            bgcolor='#f0f4f8',
        ),
        paper_bgcolor='#f0f4f8',
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.01,
            xanchor='left', x=0,
            font=dict(color='#111111', size=10),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#cccccc',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=580,
    )
    return fig


def build_3d_building_figure(floor_grids, sim_floor,
                              fire_time, smoke_time, current_time,
                              char_pos, trail, plan_path, start, exits,
                              fire_preview=None,
                              char_floor=None,
                              floor_trails=None,
                              multi_floor_path=None):
    """
    3개 층을 Z축으로 쌓아 한 화면에 렌더링하는 건축 스타일 3D 빌딩.
    sim_floor 층에만 화재·연기·대피자 시뮬레이션 결과를 표시한다.
    """
    import plotly.graph_objects as go

    path_set    = set(map(tuple, plan_path))    if plan_path    else set()
    trail_set   = set(map(tuple, trail))        if trail        else set()
    preview_set = set(map(tuple, fire_preview)) if fire_preview else set()
    exit_set    = {tuple(e) for e in exits}
    act_char_floor = char_floor if char_floor is not None else sim_floor

    rows_g = max(len(g)    for g in floor_grids.values())
    cols_g = max(len(g[0]) for g in floor_grids.values())

    mc_lighting  = dict(ambient=0.72, diffuse=0.88, roughness=0.5, specular=0.06)
    mc_light_pos = dict(x=200, y=-500, z=1200)

    fig = go.Figure()

    for floor_num in sorted(floor_grids.keys()):
        grid   = floor_grids[floor_num]
        z0     = FLOOR_BASES[floor_num]
        is_sim = (floor_num == sim_floor)
        rows   = len(grid)
        cols   = len(grid[0])

        bkt = {k: [] for k in
               ('wall', 'floor', 'room', 'stair', 'exit', 'elev', 'start',
                'path', 'trail', 'fire', 'smoke', 'prev')}

        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                pos  = (r, c)
                if cell == '#':
                    bkt['wall'].append(pos)
                elif (is_sim and fire_time is not None
                      and fire_time[r][c] != -1
                      and fire_time[r][c] <= current_time):
                    bkt['fire'].append(pos)
                elif (is_sim and smoke_time is not None
                      and smoke_time[r][c] != -1
                      and smoke_time[r][c] <= current_time):
                    bkt['smoke'].append(pos)
                elif is_sim and pos in preview_set:
                    bkt['prev'].append(pos)
                elif is_sim and pos in path_set:
                    bkt['path'].append(pos)
                elif (floor_trails and pos in {tuple(p) for p in floor_trails.get(floor_num, [])}) \
                        or (not floor_trails and is_sim and pos in trail_set):
                    bkt['trail'].append(pos)
                elif pos in exit_set or cell == 'X':
                    bkt['exit'].append(pos)
                elif cell == 'T':
                    bkt['stair'].append(pos)
                elif cell == 'S' or (is_sim and start is not None
                                     and pos == tuple(start)):
                    bkt['start'].append(pos)
                elif cell == 'E':
                    bkt['elev'].append(pos)
                elif cell == 'R':
                    bkt['room'].append(pos)
                else:
                    bkt['floor'].append(pos)

        # ── 높이 설계 ──────────────────────────────────────────────────────────
        # 벽(2.80) 반투명(0.28) → 내부 화재(1.80)·연기(1.40)가 측면 투시로 보임
        # 화재·연기는 벽보다 낮지만, 벽이 반투명이라 비침
        # (key, z_top_rel, color, opacity, legend_name, show_legend)
        spec = [
            ('floor',  FLOOR_T, AC['floor'], 1.00, f'{floor_num}층 복도',       False),
            ('room',   FLOOR_T, AC['room'],  1.00, f'{floor_num}층 강의실',     False),
            ('trail',  0.30,    AC['trail'], 0.95, '지나온 길',                 is_sim),
            ('path',   0.45,    AC['path'],  1.00, '예정 경로',                 is_sim),
            ('prev',   0.55,    AC['prev'],  1.00, '화재 예정지',               is_sim),
            ('exit',   0.60,    AC['exit'],  1.00, f'{floor_num}층 비상구',     True),
            ('start',  0.60,    AC['start'], 1.00, '시작(S)',                   is_sim),
            ('elev',   0.60,    AC['elev'],  1.00, f'{floor_num}층 엘리베이터', False),
            ('smoke',  1.40,    AC['smoke'], 0.70, '연기',                      is_sim),
            ('fire',   1.80,    AC['fire'],  1.00, '화재',                      is_sim),
            ('stair',  STAIR_H, AC['stair'], 1.00, f'{floor_num}층 계단',      True),
            ('wall',   WALL_H,  AC['wall'],  0.28, f'{floor_num}층 벽',         False),
        ]

        for key, z_top_rel, color, opacity, name, show_leg in spec:
            pts = bkt[key]
            if not pts:
                continue
            # 벽만 작은 기둥(28%)으로 렌더링 → 화재·연기 확산 시야 확보
            frac = 0.28 if key == 'wall' else 1.0
            vx, vy, vz, ti, tj, tk = _make_box_mesh(
                pts, z_bottom=z0, z_top=z0 + z_top_rel, cell_frac=frac
            )
            fig.add_trace(go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                color=color, opacity=opacity,
                name=name, showlegend=show_leg,
                legendgroup=key,
                flatshading=True,
                lighting=mc_lighting,
                lightposition=mc_light_pos,
            ))

        # 예정 경로 라인 (화재보다 살짝 위)
        if is_sim and plan_path and len(plan_path) >= 2:
            fig.add_trace(go.Scatter3d(
                x=[p[1] + 0.5 for p in plan_path],
                y=[p[0] + 0.5 for p in plan_path],
                z=[z0 + 0.50] * len(plan_path),
                mode='lines',
                line=dict(color='#ffcc00', width=5),
                name='경로 선', showlegend=False,
            ))

        # 대피자 마커 — 현재 층(act_char_floor)에만 표시
        if floor_num == act_char_floor and char_pos is not None:
            cr, cc = char_pos
            z_person = z0 + WALL_H * 0.55   # 벽 높이 55% → 반투명 벽 안에서 잘 보임
            # 배경 구체 — 이모지가 렌더 안 돼도 항상 보임
            fig.add_trace(go.Scatter3d(
                x=[cc + 0.5], y=[cr + 0.5], z=[z_person],
                mode='markers',
                marker=dict(size=20, color='#ff44cc',
                            line=dict(color='white', width=3)),
                name='대피자', showlegend=True,
            ))
            # 이모지 텍스트 — 구체 위에 겹쳐 표시
            fig.add_trace(go.Scatter3d(
                x=[cc + 0.5], y=[cr + 0.5], z=[z_person],
                mode='text',
                text=['🚶'],
                textfont=dict(size=32),
                textposition='middle center',
                showlegend=False,
            ))

        # 층 이름 레이블
        fig.add_trace(go.Scatter3d(
            x=[cols_g / 2], y=[-4.0], z=[z0 + WALL_H / 2],
            mode='text',
            text=[f'<b>{floor_num}층</b>'],
            textfont=dict(size=16, color='#333333'),
            showlegend=False,
        ))

    # ── 다중 층 전체 경로 미리보기 ─────────────────────────────────────────────
    # multi_floor_path = [(floor_num, r, c), ...]
    # 각 층 내 선분 + 계단 전환 수직 구간을 금색/주황 라인으로 표시
    if multi_floor_path and len(multi_floor_path) >= 2:
        # 층 내부 선분
        seg_start = 0
        for i in range(1, len(multi_floor_path) + 1):
            end_of_seg = (i == len(multi_floor_path)
                          or multi_floor_path[i][0] != multi_floor_path[i - 1][0])
            if end_of_seg and i > seg_start:
                seg = multi_floor_path[seg_start:i]
                fnum = seg[0][0]
                z0s  = FLOOR_BASES.get(fnum, 0)
                fig.add_trace(go.Scatter3d(
                    x=[p[2] + 0.5 for p in seg],
                    y=[p[1] + 0.5 for p in seg],
                    z=[z0s + 0.55] * len(seg),
                    mode='lines',
                    line=dict(color='#ffe066', width=6),
                    name='전체 대피 경로', showlegend=(seg_start == 0),
                    legendgroup='multi_path',
                ))
                seg_start = i - 1  # 다음 세그먼트는 전환점부터

        # 계단 전환 수직 구간
        for i in range(1, len(multi_floor_path)):
            prev_n = multi_floor_path[i - 1]
            curr_n = multi_floor_path[i]
            if prev_n[0] != curr_n[0]:
                z_top_f = FLOOR_BASES.get(prev_n[0], 0) + 0.55
                z_bot_f = FLOOR_BASES.get(curr_n[0], 0) + 0.55
                fig.add_trace(go.Scatter3d(
                    x=[prev_n[2] + 0.5, curr_n[2] + 0.5],
                    y=[prev_n[1] + 0.5, curr_n[1] + 0.5],
                    z=[z_top_f, z_bot_f],
                    mode='lines',
                    line=dict(color='#ff9900', width=8, dash='dash'),
                    name='계단 하강', showlegend=False,
                ))

    total_z = FLOOR_BASES[5] + STAIR_H + 0.1
    fig.update_layout(
        uirevision='building_3d',   # 최상위에도 설정 → Plotly.js 카메라 유지
        scene=dict(
            uirevision='building_3d',
            xaxis=dict(visible=False, range=[-1, cols_g + 1]),
            yaxis=dict(visible=False, range=[rows_g + 5, -6]),
            zaxis=dict(visible=False, range=[0, total_z]),
            aspectmode='manual',
            aspectratio=dict(
                x=cols_g / max(rows_g, cols_g) * 0.85,
                y=rows_g / max(rows_g, cols_g) * 0.85,
                z=0.42,
            ),
            bgcolor='#f8f8f8',
        ),
        paper_bgcolor='#f0f4f8',
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.01,
            xanchor='left', x=0,
            font=dict(color='#111111', size=10),
            bgcolor='rgba(255,255,255,0.90)',
            bordercolor='#cccccc',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=720,
    )
    return fig


def simulate_evacuees(grid_base, fire_time, exits, current_time, n=10):
    """
    랜덤 시작점 n명을 생성해 각자 최단 경로를 탐색하고
    evacuation 모듈로 성공/탈출시간을 기록한다.
    반환: evacuees dict, 성공 탈출시간 정렬 리스트

    자료구조: 해시맵(dict) — 대피자 정보
    알고리즘: A*(경로 탐색), 퀵 정렬(탈출 시간 정렬)
    """
    candidates = [
        (r, c)
        for r, row in enumerate(grid_base)
        for c, cell in enumerate(row)
        if cell in PASSABLE_START
    ]
    picks = random.sample(candidates, min(n, len(candidates)))

    evacuee_list = [(i + 1, f"대피자{i + 1}", pos) for i, pos in enumerate(picks)]
    evacuees = create_evacuee_dict(evacuee_list)

    for eid, data in evacuees.items():
        start = data["start_pos"]
        path, _dist, _exit, _vis = find_best_exit(
            grid_base, fire_time, start, exits, current_time
        )
        evacuees = update_evacuee_result(evacuees, eid, path or [], fire_time)

    escape_times = [
        d["escape_time"] for d in evacuees.values()
        if d["success"] and d["escape_time"] is not None
    ]
    sorted_times = quick_sort_escape_times(escape_times)
    return evacuees, sorted_times


def advance_simulation(grid_base, fire_time, exits):
    """
    실시간 재경로 + 다중 층 지원.
    매 스텝마다 현재 위치·화재 기준으로 A*를 다시 돌려 한 칸 이동.
    계단(T) 도달 시 아래 층으로 자동 전환.

    자료구조: 딕셔너리(session_state), 집합(set) — 출구/계단 빠른 조회
    알고리즘: A*(find_best_exit) — 매 스텝 재경로 탐색
    """
    import streamlit as st

    t              = st.session_state["sim_time"]
    pos            = tuple(st.session_state["char_pos"])
    selected_floor = st.session_state.get("selected_floor", 5)
    char_floor     = st.session_state.get("char_floor") or selected_floor
    exit_set       = {tuple(e) for e in exits}
    stair_set      = {tuple(s) for s in STAIR_POSITIONS}

    # ── 현재 층 그리드 / 화재 시간 결정 ──
    floor_grids_all = load_all_floor_grids()
    cur_grid = floor_grids_all.get(char_floor, grid_base)
    rows, cols = len(cur_grid), len(cur_grid[0])
    if char_floor == selected_floor:
        cur_fire = fire_time
    else:
        cur_fire = [[float('inf')] * cols for _ in range(rows)]

    # 출구 도착 → 탈출 성공
    if pos in exit_set:
        st.session_state["char_state"] = "escaped"
        st.session_state["playing"] = False
        return

    # 화재에 휩싸임 (발화 층에서만 체크)
    if char_floor == selected_floor and fire_time[pos[0]][pos[1]] <= t:
        st.session_state["char_state"] = "trapped"
        st.session_state["playing"] = False
        return

    # 탐색 대상: 실제 비상구 + 계단(3층 위 층에서만)
    extended_exits = list(exits) + list(stair_set) if char_floor > 3 else list(exits)

    # ── 1) 안전 경로 탐색 ──
    path, dist, best_exit, visited = find_best_exit(
        cur_grid, cur_fire, pos, extended_exits, t
    )
    if path and len(path) >= 2:
        st.session_state["plan_path"] = path
        st.session_state["plan_dist"] = dist
        st.session_state["best_exit"] = best_exit
        st.session_state["visited_n"] = len(visited) if visited else 0
        st.session_state["risky"] = False
        nxt = tuple(path[1])

        # 연기 지연 (발화 층에서만)
        if char_floor == selected_floor:
            smoke_time    = st.session_state.get("smoke_time")
            smoke_waiting = st.session_state.get("smoke_waiting", False)
            if (smoke_time is not None
                    and smoke_time[pos[0]][pos[1]] != -1
                    and smoke_time[pos[0]][pos[1]] <= t):
                if not smoke_waiting:
                    st.session_state["smoke_slow_count"] += 1
                    st.session_state["smoke_waiting"] = True
                    st.session_state["sim_time"] = t + 1
                    st.session_state["risky"] = True
                    return
                else:
                    st.session_state["smoke_waiting"] = False
                    st.session_state["risky"] = False

        st.session_state["char_pos"] = nxt
        st.session_state["trail"].append(nxt)
        st.session_state["sim_time"] = t + 1

        # ── 계단 도달 → 아래 층 전환 ──
        if nxt in stair_set and char_floor > 3:
            new_floor = char_floor - 1
            st.session_state["char_floor"] = new_floor
            floor_trails = st.session_state.get("floor_trails", {})
            floor_trails[char_floor] = list(st.session_state["trail"])
            floor_trails.setdefault(new_floor, [nxt])
            st.session_state["floor_trails"] = floor_trails
            st.session_state["trail"] = floor_trails[new_floor]

        if nxt in exit_set:
            st.session_state["char_state"] = "escaped"
            st.session_state["playing"] = False
        return

    if path and tuple(path[-1]) in exit_set:
        st.session_state["char_state"] = "escaped"
        st.session_state["playing"] = False
        return

    # ── 2) 안전 경로 없음 → best-effort ──
    st.session_state["plan_path"] = []
    st.session_state["best_exit"] = best_exit
    st.session_state["visited_n"] = 0
    st.session_state["risky"] = True

    trail = st.session_state["trail"]
    prev  = tuple(trail[-2]) if len(trail) >= 2 else None
    nxt   = _best_effort_next(cur_grid, cur_fire, pos, extended_exits, t, prev=prev)
    if nxt is None:
        st.session_state["char_state"] = "trapped"
        st.session_state["playing"] = False
        return

    st.session_state["char_pos"] = nxt
    trail.append(nxt)
    st.session_state["sim_time"] = t + 1

    if nxt in stair_set and char_floor > 3:
        new_floor = char_floor - 1
        st.session_state["char_floor"] = new_floor
        floor_trails = st.session_state.get("floor_trails", {})
        floor_trails[char_floor] = list(trail)
        floor_trails.setdefault(new_floor, [nxt])
        st.session_state["floor_trails"] = floor_trails
        st.session_state["trail"] = floor_trails[new_floor]

    if nxt in exit_set:
        st.session_state["char_state"] = "escaped"
        st.session_state["playing"] = False


def _best_effort_next(grid, fire_time, pos, exits, t, prev=None):
    """
    안전 경로가 없을 때의 한 칸 선택. 현재 불타지 않는 이웃 칸 중
    가장 가까운 출구 방향으로 전진한다. 직전 칸으로의 즉시 후퇴는
    다른 선택지가 있으면 피한다. 갈 곳이 없으면 None(=고립).
    """
    rows, cols = len(grid), len(grid[0])
    r, c = pos

    def exit_dist(rr, cc):
        return min(abs(rr - er) + abs(cc - ec) for er, ec in exits)

    candidates = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if grid[nr][nc] == '#':
            continue
        if fire_time[nr][nc] != -1 and fire_time[nr][nc] <= t:   # -1=미도달(안전)
            continue
        candidates.append((nr, nc))

    if not candidates:
        return None

    non_back = [p for p in candidates if prev is None or p != prev]
    pool = non_back if non_back else candidates
    pool.sort(key=lambda p: exit_dist(*p))
    return pool[0]


def _init_sim_state(st):
    """시뮬레이션 관련 session_state 기본값 1회 초기화."""
    defaults = {
        "selected_floor": 5,    # 현재 층 (3, 4, 5)
        "start": (50, 5),       # 기본 시작 위치(출구와 연결된 통로)
        "phase": "setup",       # setup(설정) | running(재생)
        "fire_mode": "랜덤",     # 랜덤 | 직접 지정
        "click_target": "내 위치",  # 직접 지정 모드에서 클릭이 무엇을 찍을지
        "manual_fires": [],     # 수동 화재 발화점 목록
        "playing": False,       # 자동 재생 여부
        "speed": 0.25,          # 한 스텝당 대기(초)
        "sim_time": 0,          # 경과 시각 t
        "char_pos": None,       # 대피자 현재 위치
        "char_state": "moving",  # moving | escaped | trapped
        "trail": [],            # 지나온 자취
        "plan_path": [],        # 현재 예정 경로
        "plan_dist": INF,
        "best_exit": None,
        "visited_n": 0,
        "risky": False,         # 안전 경로 없이 위험 이동(best-effort) 중인지
        "fire_time": None,      # 확산 시뮬레이션 결과
        "fire_positions": [],
        "evac_total": 0,        # 통계(시작 시 1회 고정)
        "evac_sorted": [],
        "smoke_time": None,
        "smoke_slow_count": 0,
        "smoke_waiting": False,  # 연기 지연 1턴 대기 중인지
        "last_click": None,     # 마지막으로 '처리한' 지도 클릭값 (중복 처리 방지)
        "fire_speed": 0.4,      # 화재 확산 속도 (1.0=현재, 0.4=느림)
        "3d_initialized": False,  # 3D 카메라 최초 설정 여부
        # ── 다중 층 대피 ──
        "char_floor": None,       # 대피자 현재 층 (None=미시작)
        "floor_trails": {},       # {floor_num: [(r,c),...]} 층별 이동 자취
        "multi_floor_path": [],   # [(floor_num,r,c),...] 전체 경로 미리보기
        "evac_paths": [],         # 히트맵용 대피자 경로 목록
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _start_simulation(st, grid_base, exits):
    """
    화재를 확정하고 재생 단계로 진입 — 통계도 이때 1회만 계산해 고정.
    알고리즘: BFS(화재·연기 확산), 다중 층 A*(전체 경로 미리보기), 퀵 정렬(통계)
    자료구조: 2D 배열(fire_time, smoke_time), 딕셔너리(session_state)
    """
    if st.session_state["fire_mode"] == "랜덤":
        positions = get_random_fire_positions(
            grid_base, count=st.session_state.get("fire_count", 2)
        )
        # 시작 위치가 화재 발화점과 겹치지 않도록 보장
        fire_pos_set = {tuple(p) for p in positions}
        for _ in range(20):
            candidate = pick_random_start(grid_base)
            if tuple(candidate) not in fire_pos_set:
                st.session_state["start"] = candidate
                break
        else:
            st.session_state["start"] = pick_random_start(grid_base)
    else:
        positions = list(st.session_state["manual_fires"])

    if not positions:
        st.warning("화재 발화점이 없습니다. 화재 지점을 1곳 이상 지정하세요.")
        return

    # 시작 위치가 화재 발화점과 겹치면 경고 (직접 지정 모드 방어)
    current_start = tuple(st.session_state["start"])
    if current_start in {tuple(p) for p in positions}:
        st.warning("시작 위치가 화재 발화점과 겹칩니다. 다른 위치를 선택하세요.")
        return

    # 실제 속도 기반 고정값
    # 셀 크기 ~3m, 사람 대피속도 ~2.5 m/s → 1셀/턴
    # 화재 복도 확산: ~0.4 m/s → 7.5턴/셀 → fire_speed = 1/7.5 ≈ 0.13
    # 연기 복도 확산: ~5 m/s → 0.6턴/셀 → smoke_speed = 1/0.6 ≈ 1.7
    FIRE_SPEED  = 0.13   # 사람보다 ~7.5배 느림
    SMOKE_SPEED = 1.7    # 사람보다 ~1.7배 빠름
    fire_time, _ = simulate_fire_spread(grid_base, positions, fire_speed=FIRE_SPEED)
    smoke_time = simulate_smoke_spread(grid_base, positions, smoke_speed=SMOKE_SPEED)
    
    st.session_state["fire_time"] = fire_time
    st.session_state["fire_positions"] = positions
    st.session_state["smoke_time"] = smoke_time
    st.session_state["phase"] = "running"
    st.session_state["sim_time"] = 0
    st.session_state["char_pos"] = st.session_state["start"]
    st.session_state["char_state"] = "moving"
    st.session_state["trail"] = [st.session_state["start"]]
    st.session_state["plan_path"] = []
    st.session_state["playing"] = True

    # ── 다중 층 초기화 ──
    sim_floor = st.session_state["selected_floor"]
    st.session_state["char_floor"] = sim_floor
    st.session_state["floor_trails"] = {sim_floor: [st.session_state["start"]]}

    # 다중 층 A* 경로 미리보기 (t=0 기준, 화재 회피 포함)
    floor_grids_all = load_all_floor_grids()
    fire_times_dict  = {sim_floor: fire_time}
    multi_path = find_multi_floor_path(
        floor_grids_all, fire_times_dict, STAIR_POSITIONS,
        sim_floor, st.session_state["start"], current_time=0,
    )
    st.session_state["multi_floor_path"] = multi_path or []

    # 대피 통계는 시작 시 한 번만 계산해 고정(매 프레임 재추첨 방지)
    _evac, sorted_times = simulate_evacuees(grid_base, fire_time, exits, 0, n=10)
    st.session_state["evac_total"] = 10
    st.session_state["evac_sorted"] = sorted_times
    # 히트맵용 경로 저장
    st.session_state["evac_paths"] = [
        d.get("path", []) for d in _evac.values() if d.get("path")
    ]


def _reset_simulation(st):
    """재생 상태만 초기화하고 설정(시작점/화재 모드)은 유지."""
    st.session_state["phase"] = "setup"
    st.session_state["playing"] = False
    st.session_state["sim_time"] = 0
    st.session_state["char_pos"] = None
    st.session_state["char_state"] = "moving"
    st.session_state["trail"] = []
    st.session_state["plan_path"] = []
    st.session_state["fire_time"] = None
    st.session_state["smoke_time"] = None
    st.session_state["smoke_waiting"] = False
    st.session_state["smoke_slow_count"] = 0
    st.session_state["char_floor"] = None
    st.session_state["floor_trails"] = {}
    st.session_state["multi_floor_path"] = []
    st.session_state["evac_paths"] = []


def render_streamlit_ui():
    import time
    import streamlit as st
    from streamlit_image_coordinates import streamlit_image_coordinates

    st.set_page_config(page_title="재난 대피 경로 시뮬레이터", layout="wide")
    st.title("🚨 재난 대피 경로 안내 시스템")
    st.caption("가천대학교 AI공학관 — 화재 확산 시 최단 탈출 경로 실시간 시뮬레이터")

    _init_sim_state(st)

    # 층 선택 — 변경 시 시뮬레이션 초기화
    floor_options = [3, 4, 5]
    selected_floor = st.session_state["selected_floor"]
    new_floor = st.sidebar.radio(
        "🏢 층 선택", floor_options,
        index=floor_options.index(selected_floor),
        format_func=lambda x: f"{x}층",
        horizontal=True,
        key="floor_radio",
    )
    if new_floor != selected_floor:
        st.session_state["selected_floor"] = new_floor
        _reset_simulation(st)
        st.rerun()

    grid_base, exits = load_base_map(st.session_state["selected_floor"])
    phase = st.session_state["phase"]
    start = st.session_state["start"]

    # ── 사이드바: 설정 / 재생 컨트롤 ──
    with st.sidebar:
        st.header(f"⚙️ {st.session_state['selected_floor']}층 시뮬레이션 설정")

        sr, sc = start
        st.subheader("📍 내 위치 (S)")
        st.success(f"행 {sr + 1}, 열 {sc + 1}")

        if phase == "setup":
            st.caption("👉 오른쪽 지도에서 통로/강의실 칸을 클릭하면 위치가 바뀝니다.")

            st.subheader("🔥 화재 설정")
            st.session_state["fire_mode"] = st.radio(
                "발화 방식", ["랜덤", "직접 지정"],
                index=0 if st.session_state["fire_mode"] == "랜덤" else 1,
                horizontal=True,
            )
            if st.session_state["fire_mode"] == "랜덤":
                st.session_state["fire_count"] = st.number_input(
                    "화재 지점 수", 1, 5, value=st.session_state.get("fire_count", 2),
                    step=1,
                )
            else:
                st.session_state["click_target"] = st.radio(
                    "지도 클릭 시 찍을 대상", ["내 위치", "화재 지점"],
                    index=0 if st.session_state["click_target"] == "내 위치" else 1,
                    horizontal=True,
                )
                st.caption(f"지정한 화재 발화점: {st.session_state['manual_fires']}")
                if st.button("화재 지점 초기화", use_container_width=True):
                    st.session_state["manual_fires"] = []
                    st.rerun()
            st.caption("🔥 화재 복도 확산 ~0.4 m/s | 💨 연기 ~5 m/s (사람 2.5 m/s 기준)")

            st.divider()
            if st.button("▶️ 시뮬레이션 시작", type="primary",
                         use_container_width=True):
                _start_simulation(st, grid_base, exits)
                st.rerun()

        else:  # running
            st.subheader("⏱️ 경과 시간")
            st.metric("t", st.session_state["sim_time"])
            st.session_state["speed"] = st.slider(
                "재생 속도", 0.1, 1.5, st.session_state["speed"], 0.1
            )
            st.caption("👈 빠름  ·  느림 👉")

            moving = st.session_state["char_state"] == "moving"
            b1, b2 = st.columns(2)
            if moving:
                label = "⏸️ 일시정지" if st.session_state["playing"] else "▶️ 재생"
                if b1.button(label, use_container_width=True):
                    st.session_state["playing"] = not st.session_state["playing"]
                    st.rerun()
                if b2.button("⏭️ 다음 단계", use_container_width=True,
                             disabled=st.session_state["playing"]):
                    advance_simulation(grid_base, st.session_state["fire_time"], exits)
                    st.rerun()
            if st.button("🔄 처음으로", use_container_width=True):
                _reset_simulation(st)
                st.rerun()

    def _main_view():
        import time as _time
        from streamlit_image_coordinates import streamlit_image_coordinates as _sic

        _floor  = st.session_state.get("selected_floor", 5)
        _phase  = st.session_state.get("phase", "setup")
        _start  = st.session_state.get("start", (50, 5))
        _fmode  = st.session_state.get("fire_mode", "랜덤")
        _grid_base, _exits = load_base_map(_floor)
        _rows, _cols = len(_grid_base), len(_grid_base[0])
        _CELL_PX = 12

        if _phase == "running":
            _t          = st.session_state.get("sim_time", 0)
            _char_pos   = st.session_state.get("char_pos")
            _trail      = st.session_state.get("trail") or []
            _plan_path  = st.session_state.get("plan_path") or []
            _fire_time  = st.session_state.get("fire_time")
            _smoke_time = st.session_state.get("smoke_time")
            _fire_preview = None
        else:
            _t = 0
            _char_pos = _trail = _plan_path = _fire_time = _smoke_time = None
            _fire_preview = (st.session_state.get("manual_fires")
                             if _fmode == "직접 지정" else None)

        _fire_time_2d = (_fire_time if _fire_time is not None
                         else [[INF] * _cols for _ in range(_rows)])
        _grid = make_grid_with_start(_grid_base, _start)

        # 3D 차트
        _char_floor       = st.session_state.get("char_floor") or _floor
        _floor_trails     = st.session_state.get("floor_trails", {})
        _multi_floor_path = st.session_state.get("multi_floor_path", [])
        _floor_label = (f"{_char_floor}층 이동 중" if _char_floor != _floor
                        else f"{_floor}층 시뮬레이션 중")
        st.subheader(f"🏢 3D 빌딩 뷰  ({_floor_label}, t = {_t})")
        _fgrids = load_all_floor_grids()
        _fig = build_3d_building_figure(
            _fgrids, _floor,
            _fire_time, _smoke_time, _t,
            _char_pos, _trail, _plan_path, _start, _exits,
            fire_preview=_fire_preview,
            char_floor=_char_floor,
            floor_trails=_floor_trails,
            multi_floor_path=_multi_floor_path,
        )
        if not st.session_state.get("3d_initialized"):
            _fig.update_layout(scene_camera=dict(
                eye=dict(x=1.6, y=-2.2, z=0.9),
                up=dict(x=0, y=0, z=1),
            ))
            st.session_state["3d_initialized"] = True
        st.plotly_chart(_fig, use_container_width=True, key="plotly_3d_building")

        # 2D 지도 + 통계
        col_map, col_info = st.columns([3, 2])

        with col_map:
            with st.expander(
                f"🗺️ 2D 지도  (클릭으로 위치·화재 설정) — {_floor}층",
                expanded=True,
            ):
                pil = build_map_image(
                    _grid, _fire_time_2d, _plan_path, _start, _exits, _t, _CELL_PX,
                    char_pos=_char_pos, trail=_trail, fire_preview=_fire_preview,
                    trail_on_top=(st.session_state.get("char_state") == "escaped"),
                    smoke_time=_smoke_time,
                )
                coords = _sic(pil, key="map_click")
                st.markdown(legend_html(), unsafe_allow_html=True)

                if _phase == "setup":
                    st.caption("통로(흰색)·강의실(R) 칸을 클릭해 위치를 설정하세요. "
                               "'직접 지정' 모드에선 사이드바에서 클릭 대상을 선택합니다.")
                    sig = tuple(sorted(coords.items())) if coords is not None else None
                    if sig is not None and sig != st.session_state.get("last_click"):
                        st.session_state["last_click"] = sig
                        cc = min(max(int(coords["x"] / coords["width"] * _cols), 0), _cols - 1)
                        cr = min(max(int(coords["y"] / coords["height"] * _rows), 0), _rows - 1)
                        clicked = (cr, cc)
                        manual_fire = (_fmode == "직접 지정"
                                       and st.session_state.get("click_target") == "화재 지점")
                        if manual_fire:
                            if _grid_base[cr][cc] in FIRE_OK:
                                fires = st.session_state["manual_fires"]
                                if clicked in fires:
                                    fires.remove(clicked)
                                else:
                                    fires.append(clicked)
                                st.rerun()
                            else:
                                st.warning(f"({cr + 1}, {cc + 1})은(는) 화재를 둘 수 없는 칸입니다.")
                        elif clicked != _start:
                            if _grid_base[cr][cc] in PASSABLE_START:
                                st.session_state["start"] = clicked
                                st.rerun()
                            else:
                                st.warning(f"({cr + 1}, {cc + 1})은(는) 이동할 수 없는 칸입니다. "
                                           "통로/강의실을 클릭하세요.")
                else:
                    st.caption("▶️ 재생 중에는 지도 클릭이 비활성화됩니다. "
                               "위치·화재를 바꾸려면 '처음으로'를 누르세요.")

        with col_info:
            if _phase == "setup":
                st.info("👈 위치와 화재를 설정한 뒤 **시뮬레이션 시작**을 누르세요.")
                st.markdown(
                    "- **랜덤**: 화재 지점 수만 정하면 무작위로 발화\n"
                    "- **직접 지정**: 실제 상황처럼 화재 위치를 클릭으로 지정\n"
                    "- 시작하면 대피자가 **매 순간 화재를 피해 경로를 다시 잡으며** 탈출합니다."
                )
                # 자동 재생 루프 없음 — setup 단계이므로 여기서 종료
                return

            st.subheader("🧭 대피 진행")
            _state     = st.session_state.get("char_state", "moving")
            _t_now     = st.session_state.get("sim_time", 0)
            _cf        = st.session_state.get("char_floor") or _floor
            _floors_done = sorted(
                [f for f in st.session_state.get("floor_trails", {}) if f != _cf],
                reverse=True,
            )
            _floor_info = f"현재 {_cf}층"
            if _floors_done:
                _floor_info += f"  (통과: {' → '.join(str(f) for f in _floors_done)}층)"
            st.caption(f"🏢 {_floor_info}")
            if _state == "escaped":
                st.success(f"탈출 성공 ✅ — {_t_now}초 만에 비상구 도착")
            elif _state == "trapped":
                st.error(f"탈출 실패 ❌ — 화재에 가로막힘 (t={_t_now})")
            elif st.session_state.get("risky"):
                st.warning("⚠️ 안전 경로 없음 — 화재를 피해 탈출 시도 중…")
            else:
                remain = max(len(st.session_state.get("plan_path") or []) - 1, 0)
                st.info(f"🟣 대피 중 — 출구까지 약 {remain}칸 남음")

            best_exit = st.session_state.get("best_exit")
            m1, m2, m3 = st.columns(3)
            m1.metric("이동한 거리", f"{max(len(st.session_state.get('trail') or []) - 1, 0)} 칸")
            m2.metric("목표 출구", f"{best_exit}" if best_exit else "—")
            if best_exit:
                fa = st.session_state["fire_time"][best_exit[0]][best_exit[1]]
                m3.metric("출구 화재 도달", "안전" if fa in (-1, INF) else f"t={int(fa)}")
            else:
                m3.metric("출구 화재 도달", "—")
            st.caption(f"이번 스텝 A* 탐색 노드: {st.session_state.get('visited_n', 0)}개  ·  "
                       f"화재 발화점: {st.session_state.get('fire_positions', [])}")
            m4, m5 = st.columns(2)
            m4.metric("연기 지연", f"{st.session_state.get('smoke_slow_count', 0)}회")
            m5.metric("연기 영향",
                      "있음" if st.session_state.get("smoke_slow_count", 0) > 0 else "없음")

            risk_score = calculate_path_risk(
                st.session_state.get("plan_path") or [],
                st.session_state.get("fire_time"),
                st.session_state.get("sim_time", 0),
            )
            risk_level = get_risk_level(risk_score)
            st.divider()
            st.subheader("⚠️ 현재 경로 위험도")
            r1, r2 = st.columns(2)
            r1.metric("위험도 점수", f"{risk_score:.1f}")
            r2.metric("안전 등급", risk_level)
            if risk_level == "위험":
                st.error("현재 경로는 화재 도달 시간이 가까워 위험합니다.")
            elif risk_level == "주의":
                st.warning("현재 경로는 주의가 필요합니다.")
            else:
                st.success("현재 경로는 비교적 안전합니다.")

            st.divider()
            st.subheader("📊 전체 대피 통계 (랜덤 10명)")
            st.caption("시작 시점에 무작위 10명을 배치해 각자 최단 경로로 탈출시킨 결과입니다.")
            sorted_times = st.session_state.get("evac_sorted", [])
            total = st.session_state.get("evac_total", 0)
            success = len(sorted_times)
            target_time = st.slider("목표 탈출 시간(초)", 0, 50, value=20, key="target_t")
            st.caption("👆 이 시간 안에 몇 명이 빠져나갔는지(이진 탐색) 확인하는 기준입니다.")
            under = binary_search_time(sorted_times, target_time)
            avg = sum(sorted_times) / success if success else 0.0
            s1, s2, s3 = st.columns(3)
            s1.metric("탈출 성공", f"{success}/{total}")
            s2.metric("평균 탈출시간", f"{avg:.1f}" if success else "—")
            s3.metric(f"t≤{target_time} 인원", f"{under}명")
            st.caption(f"탈출시간 정렬(퀵정렬): {sorted_times}")
            st.divider()
            st.subheader("🚪 출구 혼잡도 분석")
            exit_capacity = st.slider(
                "출구 처리 용량(시간당 인원)", 1, 5, value=1, step=1, key="exit_capacity"
            )
            st.caption("출구에 동시에 여러 명이 도착하면, 설정한 처리 용량만큼만 탈출하고 나머지는 대기합니다.")
            evac_cong, _ = simulate_evacuees(
                _grid_base, st.session_state["fire_time"], _exits, 0, n=10
            )
            cong_stats = get_evacuation_statistics(
                evac_cong, target_time=target_time,
                use_congestion=True, exit_capacity=exit_capacity,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("최대 대기 인원", f"{cong_stats['max_waiting']}명")
            c2.metric("평균 대기 시간", f"{cong_stats['average_waiting_time']:.1f}초")
            c3.metric("대기 발생 인원", f"{cong_stats['total_congested_people']}명")

            # ── 위험도 히트맵 ──────────────────────────────────────────────────
            st.divider()
            with st.expander("🌡️ 위험도 히트맵", expanded=False):
                st.caption(
                    "🔴 빨강 = 화재 빠름(위험)  🟢 초록 = 안전  "
                    "🔵 파랑 = 비상구  🟠 주황 = 계단  "
                    "● 경로 병목도 반영"
                )
                _hm_img = build_heatmap_image(
                    _grid_base,
                    _fire_time_2d,
                    evacuee_paths=st.session_state.get("evac_paths") or None,
                    cell_px=8,
                )
                st.image(_hm_img, use_container_width=True)

        # ── 자동 재생 — fragment scope 만 rerun → Plotly 절대 재마운트 없음 ──
        if (st.session_state.get("playing")
                and st.session_state.get("char_state") == "moving"
                and st.session_state.get("fire_time") is not None):
            _time.sleep(st.session_state.get("speed", 0.5))
            advance_simulation(_grid_base, st.session_state["fire_time"], _exits)
            st.rerun()

    _main_view()


def _running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        # suppress_warning: 터미널(python3 main.py) 모드의 ScriptRunContext 경고 억제
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_in_streamlit():
        render_streamlit_ui()
    else:
        run_integration_test()
