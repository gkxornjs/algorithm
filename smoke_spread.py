# ============================================================
# smoke_spread.py
# 재난 대피 경로 안내 시스템 - 연기 확산 시뮬레이션 모듈
#
# 실행 환경: VSCode / Python 3.9+
# 필요 라이브러리: 없음 (Python 기본 내장 모듈만 사용)
#
# 자료구조: 큐(deque), 2D 배열(smoke_time)
# 알고리즘: BFS(연기 확산)
#
# Input 데이터: grid(2D 배열) — map_builder.py 에서 전달받음
# ============================================================

from collections import deque

WALL = "#"
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def is_valid(grid, r, c):
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def simulate_smoke_spread(grid, fire_positions, smoke_speed=2):
    """
    BFS 기반 연기 확산 시뮬레이션.

    smoke_speed 파라미터:
      - 클수록 빠르게 확산 (step = 1 / smoke_speed)
      - 기본값 2 → 화재보다 2배 빠른 확산

    smoke_time[r][c]:
      - -1   : 연기 미도달 (벽 또는 고립 구역)
      -  0   : 발화(연기 발생) 위치
      - 양수  : 해당 시각에 연기 도달
    """
    rows = len(grid)
    cols = len(grid[0])

    # 자료구조: 2D 배열(smoke_time) — 각 셀의 연기 도달 시각
    smoke_time = [[-1] * cols for _ in range(rows)]
    # 자료구조: 큐(deque) / 알고리즘: BFS 연기 확산
    queue = deque()
    step = 1.0 / smoke_speed   # 한 칸 확산에 걸리는 시간

    if fire_positions is None:
        fire_positions = []
    if isinstance(fire_positions, tuple):
        fire_positions = [fire_positions]

    # ── 발화 위치 초기화 ──
    for r, c in fire_positions:
        if is_valid(grid, r, c) and grid[r][c] != WALL:
            smoke_time[r][c] = 0
            queue.append((r, c))

    # ── BFS 확산 ──────────────────────────────────────────────────────────────
    # 알고리즘: BFS — 가중 거리(step) 기반 균일 확산
    while queue:
        r, c = queue.popleft()
        current_time = smoke_time[r][c]

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not is_valid(grid, nr, nc):
                continue
            if grid[nr][nc] == WALL:
                continue
            if smoke_time[nr][nc] == -1:         # 아직 미방문 셀만 처리
                smoke_time[nr][nc] = current_time + step
                queue.append((nr, nc))

    return smoke_time


def get_smoke_cells_at_time(smoke_time, t):
    """t 시각 이전에 연기가 도달한 모든 셀 반환."""
    return [
        (r, c)
        for r in range(len(smoke_time))
        for c in range(len(smoke_time[r]))
        if smoke_time[r][c] != -1 and smoke_time[r][c] <= t
    ]


def is_smoke_danger(pos, smoke_time, current_time):
    """pos 셀이 current_time 시각에 연기 위험 구역인지 반환."""
    r, c = pos
    return smoke_time[r][c] != -1 and smoke_time[r][c] <= current_time
