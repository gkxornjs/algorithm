import os
import random
import matplotlib
matplotlib.use('Agg')   # GUI 없는 환경에서 plt.show() 블로킹 방지
from map_builder import build_map
from fire_spread import simulate_fire, get_exit_fire_times
from path_finder import get_escape_path, is_escape_possible, find_best_exit, parse_map
from evacuation import (
    create_evacuee_dict,
    update_evacuee_result,
    print_evacuation_statistics,
)

MAP_FILE = os.path.join(os.path.dirname(__file__), "building_map.txt")

PASSABLE = {'.', 'R', 'X', 'E', 'S', 'F'}


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


if __name__ == "__main__":
    run_integration_test()
