# ============================================================
# evacuation.py
# 재난 대피 경로 안내 시스템 - 대피자 관리 및 통계 모듈
#
# 실행 환경: VSCode / Python 3.9+
# 필요 라이브러리: 없음
#
# 자료구조: 해시맵(dict), 스택(list), 큐(deque)
# 알고리즘: 퀵 정렬(탈출 시간 정렬), 이진 탐색(목표 시간 내 인원),
#           출구 혼잡도 시뮬레이션(FIFO 큐)
# ============================================================

from collections import deque


# ============================================================
# 대피자 딕셔너리 관리
# 자료구조: 해시맵(dict) — {evacuee_id: {name, pos, escape_time, ...}}
# ============================================================

def create_evacuee_dict(evacuee_list):
    """
    대피자 목록을 해시맵으로 초기화.
    자료구조: 해시맵(dict)
    """
    # 자료구조: 해시맵(dict) — evacuee_id를 키로 대피자 정보 저장
    evacuees = {}
    for evacuee_id, name, start_pos in evacuee_list:
        evacuees[evacuee_id] = {
            "name":         name,
            "start_pos":    start_pos,
            "current_pos":  start_pos,
            "escape_time":  None,
            "success":      False,
            "failed":       False,
            "path":         [],
            "message":      "대기 중",
            "waiting_time": 0,
        }
    return evacuees


def save_path_with_stack(path):
    """
    스택으로 경로를 역순 복원 (push/pop).
    자료구조: 스택(list) — append/pop으로 LIFO 구현
    알고리즘: 스택 기반 역순 변환
    """
    # 자료구조: 스택(list) — 경로를 push
    stack = []
    for pos in path:
        stack.append(pos)   # push

    reversed_path = []
    while stack:
        reversed_path.append(stack.pop())   # pop — LIFO
    return reversed_path


def get_evacuee_position_at_time(path, t):
    """시각 t에서 대피자 위치 반환."""
    if not path:
        return None
    if t < 0:
        return path[0]
    if t >= len(path):
        return path[-1]
    return path[t]


def check_fire_collision(path, fire_time, t):
    """시각 t에서 대피자가 화재와 충돌하는지 확인."""
    if not path:
        return False
    r, c = get_evacuee_position_at_time(path, t)
    return fire_time[r][c] != -1 and fire_time[r][c] <= t


def check_escape_result(path, fire_time):
    """경로 전체를 순회해 탈출 성공 여부와 탈출 시각 반환."""
    if not path:
        return False, None
    for step, (r, c) in enumerate(path):
        if fire_time[r][c] != -1 and fire_time[r][c] <= step:
            return False, step
    return True, len(path) - 1


def update_evacuee_position_by_time(evacuees, evacuee_id, t, fire_time):
    """시각 t 기준으로 대피자 위치와 상태를 갱신."""
    path = evacuees[evacuee_id]["path"]

    if not path:
        evacuees[evacuee_id]["message"] = "탈출 경로 없음"
        evacuees[evacuee_id]["failed"]  = True
        return evacuees

    evacuees[evacuee_id]["current_pos"] = get_evacuee_position_at_time(path, t)

    if check_fire_collision(path, fire_time, t):
        evacuees[evacuee_id].update({
            "success":     False,
            "failed":      True,
            "escape_time": None,
            "message":     f"t={t} 화재와 충돌",
        })
        return evacuees

    if t >= len(path) - 1:
        evacuees[evacuee_id].update({
            "success":     True,
            "failed":      False,
            "escape_time": len(path) - 1,
            "message":     "탈출 성공",
        })
    else:
        evacuees[evacuee_id].update({
            "success":     False,
            "failed":      False,
            "escape_time": None,
            "message":     "대피 중",
        })
    return evacuees


def update_all_evacuees_by_time(evacuees, t, fire_time):
    for evacuee_id in evacuees:
        update_evacuee_position_by_time(evacuees, evacuee_id, t, fire_time)
    return evacuees


def update_evacuee_result(evacuees, evacuee_id, path, fire_time):
    """경로 탐색 결과를 대피자 딕셔너리에 반영."""
    success, escape_time = check_escape_result(path, fire_time)
    evacuees[evacuee_id].update({
        "path":         path,
        "success":      success,
        "escape_time":  escape_time,
        "current_pos":  path[0] if path else None,
        "failed":       not success,
        "message":      "탈출 가능" if success else "탈출 실패 위험",
    })
    return evacuees


# ============================================================
# 퀵 정렬 — 탈출 시간 오름차순 정렬
# 알고리즘: 퀵 정렬 (분할 정복, 피벗 = 중간 원소)
# ============================================================

def quick_sort_escape_times(arr):
    """
    탈출 시간 리스트를 퀵 정렬로 오름차순 정렬.
    알고리즘: 퀵 정렬 (분할 정복)
    """
    if len(arr) <= 1:
        return arr

    pivot  = arr[len(arr) // 2]               # 피벗: 중간 원소 선택
    left   = [x for x in arr if x < pivot]    # 피벗 미만
    middle = [x for x in arr if x == pivot]   # 피벗과 같음
    right  = [x for x in arr if x > pivot]    # 피벗 초과

    # 알고리즘: 재귀적 분할 정복
    return quick_sort_escape_times(left) + middle + quick_sort_escape_times(right)


# ============================================================
# 이진 탐색 — 목표 시간 이하 탈출 인원 수 계산
# 알고리즘: 이진 탐색 (정렬된 배열에서 upper-bound 탐색)
# ============================================================

def binary_search_time(sorted_times, target_time):
    """
    정렬된 탈출 시간 배열에서 target_time 이하인 원소 개수 반환.
    알고리즘: 이진 탐색 (upper-bound)
    자료구조: 정렬된 배열(sorted_times)
    """
    left, right, answer = 0, len(sorted_times) - 1, -1

    while left <= right:
        mid = (left + right) // 2
        if sorted_times[mid] <= target_time:
            answer = mid    # 조건 만족 → 오른쪽 탐색 (더 많은 인원 가능)
            left = mid + 1
        else:
            right = mid - 1

    return answer + 1   # 0-indexed → 인원 수


# ============================================================
# 출구 혼잡도 시뮬레이션
# 자료구조: 큐(deque), 해시맵(dict)
# 알고리즘: FIFO 큐 기반 시간 단위 시뮬레이션
# ============================================================

def simulate_exit_congestion(evacuees, exit_capacity=1):
    """
    출구에 동시 도착 시 대기열 생성 시뮬레이션.
    exit_capacity: 시간 단위당 처리 인원 수.

    자료구조: 큐(deque) — 대기 인원 FIFO 처리
    알고리즘: FIFO 큐 기반 시간 단위 시뮬레이션
    """
    # 자료구조: 해시맵(dict) — 도착 시각별 대피자 목록
    arrival_table = {}
    for evacuee_id, data in evacuees.items():
        if data["success"] and data["escape_time"] is not None:
            t = data["escape_time"]
            arrival_table.setdefault(t, []).append(evacuee_id)

    if not arrival_table:
        return {
            "congestion_result":       {},
            "max_waiting":             0,
            "average_waiting_time":    0,
            "total_congested_people":  0,
        }

    total_arrivals = sum(len(v) for v in arrival_table.values())
    current_time   = min(arrival_table.keys())
    end_time       = max(arrival_table.keys()) + len(evacuees) + 1

    # 자료구조: 큐(deque) — FIFO 대기열
    waiting_queue  = deque()
    congestion_result = {}
    escaped_count  = 0
    max_waiting    = 0

    while current_time <= end_time:
        # 이번 시각에 도착한 대피자를 대기열에 추가
        for evacuee_id in arrival_table.get(current_time, []):
            waiting_queue.append((evacuee_id, current_time))   # enqueue

        max_waiting = max(max_waiting, len(waiting_queue))

        # 용량만큼 대기열에서 처리
        for _ in range(exit_capacity):
            if waiting_queue:
                evacuee_id, arrival_time = waiting_queue.popleft()   # dequeue
                waiting_time = current_time - arrival_time
                congestion_result[evacuee_id] = {
                    "arrival_time":      arrival_time,
                    "final_escape_time": current_time,
                    "waiting_time":      waiting_time,
                }
                evacuees[evacuee_id]["escape_time"]   = current_time
                evacuees[evacuee_id]["waiting_time"]  = waiting_time
                escaped_count += 1

        if escaped_count == total_arrivals:
            break
        current_time += 1

    waiting_times = [d["waiting_time"] for d in congestion_result.values()]
    avg_waiting   = sum(waiting_times) / len(waiting_times) if waiting_times else 0
    total_cong    = sum(1 for w in waiting_times if w > 0)

    return {
        "congestion_result":       congestion_result,
        "max_waiting":             max_waiting,
        "average_waiting_time":    avg_waiting,
        "total_congested_people":  total_cong,
    }


# ============================================================
# 통계 집계
# ============================================================

def get_evacuation_statistics(evacuees, target_time=5,
                               use_congestion=False, exit_capacity=1):
    """
    전체 대피 통계 계산.
    알고리즘: 퀵 정렬(탈출 시간 정렬), 이진 탐색(목표 시간 내 인원)
    """
    if use_congestion:
        congestion = simulate_exit_congestion(evacuees, exit_capacity)
    else:
        congestion = {
            "congestion_result":       {},
            "max_waiting":             0,
            "average_waiting_time":    0,
            "total_congested_people":  0,
        }

    escape_times = [
        data["escape_time"]
        for data in evacuees.values()
        if data["success"] and data["escape_time"] is not None
    ]

    # 알고리즘: 퀵 정렬 — 탈출 시간 오름차순 정렬
    sorted_times = quick_sort_escape_times(escape_times)
    # 알고리즘: 이진 탐색 — target_time 이하 탈출 인원 수
    count_under_target = binary_search_time(sorted_times, target_time)

    total         = len(evacuees)
    success_count = len(escape_times)
    avg_time      = sum(escape_times) / success_count if success_count > 0 else 0

    return {
        "total":                  total,
        "success_count":          success_count,
        "fail_count":             total - success_count,
        "success_rate":           success_count / total * 100 if total > 0 else 0,
        "average_escape_time":    avg_time,
        "sorted_escape_times":    sorted_times,
        "count_under_target":     count_under_target,
        "max_waiting":            congestion["max_waiting"],
        "average_waiting_time":   congestion["average_waiting_time"],
        "total_congested_people": congestion["total_congested_people"],
        "congestion_result":      congestion["congestion_result"],
    }


def print_evacuation_statistics(evacuees, target_time=5,
                                 use_congestion=False, exit_capacity=1):
    stats = get_evacuation_statistics(evacuees, target_time, use_congestion, exit_capacity)

    print("\n=== 대피 통계 ===")
    print(f"전체 대피자 수: {stats['total']}")
    print(f"탈출 성공: {stats['success_count']}")
    print(f"탈출 실패: {stats['fail_count']}")
    print(f"탈출 성공률: {stats['success_rate']:.2f}%")
    print(f"탈출 시간 정렬 결과: {stats['sorted_escape_times']}")
    print(f"t={target_time} 이하 탈출 인원: {stats['count_under_target']}")
    print(f"평균 탈출 시간: {stats['average_escape_time']:.2f}")

    if use_congestion:
        print("\n=== 출구 혼잡도 분석 ===")
        print(f"출구 처리 용량: 시간당 {exit_capacity}명")
        print(f"최대 대기 인원: {stats['max_waiting']}")
        print(f"평균 대기 시간: {stats['average_waiting_time']:.2f}")
        print(f"대기 발생 인원: {stats['total_congested_people']}")


def get_simulation_state(evacuees, fire_time, t,
                          target_time=5, use_congestion=False, exit_capacity=1):
    update_all_evacuees_by_time(evacuees, t, fire_time)
    statistics = get_evacuation_statistics(evacuees, target_time, use_congestion, exit_capacity)
    return {"time": t, "evacuees": evacuees, "statistics": statistics}
