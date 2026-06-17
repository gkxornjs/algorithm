# smoke_spread.py

from collections import deque

# 자료구조: 큐(deque)
# 자료구조: 2D 배열(smoke_time)
# 알고리즘: BFS(연기 확산)

WALL = "#"
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def is_valid(grid, r, c):
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def simulate_smoke_spread(grid, fire_positions, smoke_speed=2):
    """
    연기 확산 시뮬레이션

    smoke_speed=2이면
    화재보다 연기가 더 빠르게 퍼지는 상황을 표현한다.

    smoke_time[r][c]
    -1 : 연기 미도달
    0 이상 : 해당 시간에 연기 도달
    """

    rows = len(grid)
    cols = len(grid[0])

    smoke_time = [[-1 for _ in range(cols)] for _ in range(rows)]
    queue = deque()

    if fire_positions is None:
        fire_positions = []

    if isinstance(fire_positions, tuple):
        fire_positions = [fire_positions]

    for r, c in fire_positions:
        if is_valid(grid, r, c) and grid[r][c] != WALL:
            smoke_time[r][c] = 0
            queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        current_time = smoke_time[r][c]

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc

            if not is_valid(grid, nr, nc):
                continue

            if grid[nr][nc] == WALL:
                continue

            if smoke_time[nr][nc] == -1:
                # smoke_speed가 클수록 더 빠르게 도달
                next_time = current_time + 1 / smoke_speed
                smoke_time[nr][nc] = next_time
                queue.append((nr, nc))

    return smoke_time


def get_smoke_cells_at_time(smoke_time, t):
    smoke_cells = []

    for r in range(len(smoke_time)):
        for c in range(len(smoke_time[r])):
            if smoke_time[r][c] != -1 and smoke_time[r][c] <= t:
                smoke_cells.append((r, c))

    return smoke_cells


def is_smoke_danger(pos, smoke_time, current_time):
    r, c = pos

    if smoke_time[r][c] != -1 and smoke_time[r][c] <= current_time:
        return True

    return False