# ============================================================
# evacuation.py
# 재난 대피 경로 안내 시스템 - 대피자 관리 및 통계 모듈
#
# 실행 환경: VSCode / Python 3.9+
# 필요 라이브러리: 없음 (Python 기본 내장 모듈만 사용)
#
# 자료구조: 해시맵(dict), 스택(stack)
# 알고리즘: 퀵 정렬, 이진 탐색
#
# Input 데이터: 대피자 정보(evacuee_list), 경로(path), fire_time — 직접 구성 / main.py에서 전달받음
# ============================================================


def create_evacuee_dict(evacuee_list):
    # 자료구조: 해시맵(dict) — 대피자 ID를 키로 정보 저장
    evacuees = {}

    for evacuee_id, name, start_pos in evacuee_list:
        evacuees[evacuee_id] = {
            "name": name,
            "start_pos": start_pos,
            "current_pos": start_pos,
            "escape_time": None,
            "success": False,
            "failed": False,
            "path": [],
            "message": "대기 중"
        }

    return evacuees


def save_path_with_stack(path):
    # 자료구조: 스택(stack) — 경로 역순 저장
    stack = []

    for pos in path:
        stack.append(pos)

    reversed_path = []

    while stack:
        reversed_path.append(stack.pop())

    return reversed_path


def get_evacuee_position_at_time(path, t):
    if not path:
        return None

    if t < 0:
        return path[0]

    if t >= len(path):
        return path[-1]

    return path[t]


def check_fire_collision(path, fire_time, t):
    if not path:
        return False

    current_pos = get_evacuee_position_at_time(path, t)
    r, c = current_pos

    if fire_time[r][c] != -1 and fire_time[r][c] <= t:
        return True

    return False


def check_escape_result(path, fire_time):
    if not path:
        return False, None

    for time, pos in enumerate(path):
        r, c = pos

        if fire_time[r][c] != -1 and fire_time[r][c] <= time:
            return False, time

    return True, len(path) - 1


def update_evacuee_position_by_time(evacuees, evacuee_id, t, fire_time):
    path = evacuees[evacuee_id]["path"]

    if not path:
        evacuees[evacuee_id]["message"] = "탈출 경로 없음"
        evacuees[evacuee_id]["failed"] = True
        return evacuees

    current_pos = get_evacuee_position_at_time(path, t)
    evacuees[evacuee_id]["current_pos"] = current_pos

    if check_fire_collision(path, fire_time, t):
        evacuees[evacuee_id]["success"] = False
        evacuees[evacuee_id]["failed"] = True
        evacuees[evacuee_id]["escape_time"] = None
        evacuees[evacuee_id]["message"] = f"t={t} 화재와 충돌"
        return evacuees

    if t >= len(path) - 1:
        evacuees[evacuee_id]["success"] = True
        evacuees[evacuee_id]["failed"] = False
        evacuees[evacuee_id]["escape_time"] = len(path) - 1
        evacuees[evacuee_id]["message"] = "탈출 성공"
    else:
        evacuees[evacuee_id]["success"] = False
        evacuees[evacuee_id]["failed"] = False
        evacuees[evacuee_id]["escape_time"] = None
        evacuees[evacuee_id]["message"] = "대피 중"

    return evacuees


def update_all_evacuees_by_time(evacuees, t, fire_time):
    for evacuee_id in evacuees:
        update_evacuee_position_by_time(evacuees, evacuee_id, t, fire_time)

    return evacuees


def update_evacuee_result(evacuees, evacuee_id, path, fire_time):
    success, escape_time = check_escape_result(path, fire_time)

    evacuees[evacuee_id]["path"] = path
    evacuees[evacuee_id]["success"] = success
    evacuees[evacuee_id]["escape_time"] = escape_time
    evacuees[evacuee_id]["current_pos"] = path[0] if path else None

    if success:
        evacuees[evacuee_id]["failed"] = False
        evacuees[evacuee_id]["message"] = "탈출 가능"
    else:
        evacuees[evacuee_id]["failed"] = True
        evacuees[evacuee_id]["message"] = "탈출 실패 위험"

    return evacuees


def quick_sort_escape_times(arr):
    # 알고리즘: 퀵 정렬 — 탈출 시간 오름차순 정렬
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]

    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort_escape_times(left) + middle + quick_sort_escape_times(right)


def binary_search_time(sorted_times, target_time):
    # 알고리즘: 이진 탐색 — 목표 시간 이하 탈출 인원 수 탐색
    left = 0
    right = len(sorted_times) - 1
    answer = -1

    while left <= right:
        mid = (left + right) // 2

        if sorted_times[mid] <= target_time:
            answer = mid
            left = mid + 1
        else:
            right = mid - 1

    return answer + 1

def simulate_exit_congestion(evacuees, exit_capacity=1):
    """
    출구 혼잡도 시뮬레이션

    exit_capacity:
    - 한 시간 단위에 출구에서 탈출 가능한 인원 수
    - 기본값 1명

    동작:
    - 탈출 성공한 대피자들을 escape_time 기준으로 모음
    - 같은 시간에 출구에 도착한 인원이 많으면 대기 발생
    - 출구는 한 시간에 exit_capacity명만 처리
    """

    # 자료구조: 해시맵(dict) — 시간별 도착 대피자 관리
    arrival_table = {}

    for evacuee_id, data in evacuees.items():
        if data["success"] and data["escape_time"] is not None:
            arrival_time = data["escape_time"]

            if arrival_time not in arrival_table:
                arrival_table[arrival_time] = []

            arrival_table[arrival_time].append(evacuee_id)

    # 자료구조: 큐(list) — 출구 대기열
    waiting_queue = []

    congestion_result = {}
    escaped_count = 0
    max_waiting = 0

    if not arrival_table:
        return {
            "congestion_result": {},
            "max_waiting": 0,
            "average_waiting_time": 0,
            "total_congested_people": 0
        }

    current_time = min(arrival_table.keys())
    end_time = max(arrival_table.keys()) + len(evacuees) + 1

    while current_time <= end_time:
        # 현재 시간에 출구에 도착한 사람들을 대기열에 추가
        if current_time in arrival_table:
            for evacuee_id in arrival_table[current_time]:
                waiting_queue.append((evacuee_id, current_time))

        max_waiting = max(max_waiting, len(waiting_queue))

        # 출구 처리: 한 시간에 exit_capacity명만 탈출 가능
        for _ in range(exit_capacity):
            if waiting_queue:
                evacuee_id, arrival_time = waiting_queue.pop(0)
                final_escape_time = current_time
                waiting_time = final_escape_time - arrival_time

                congestion_result[evacuee_id] = {
                    "arrival_time": arrival_time,
                    "final_escape_time": final_escape_time,
                    "waiting_time": waiting_time
                }

                evacuees[evacuee_id]["escape_time"] = final_escape_time
                evacuees[evacuee_id]["waiting_time"] = waiting_time
                escaped_count += 1

        if escaped_count == sum(len(v) for v in arrival_table.values()):
            break

        current_time += 1

    waiting_times = [
        data["waiting_time"]
        for data in congestion_result.values()
    ]

    average_waiting_time = 0
    if waiting_times:
        average_waiting_time = sum(waiting_times) / len(waiting_times)

    total_congested_people = len([
        w for w in waiting_times if w > 0
    ])

    return {
        "congestion_result": congestion_result,
        "max_waiting": max_waiting,
        "average_waiting_time": average_waiting_time,
        "total_congested_people": total_congested_people
    }

def get_evacuation_statistics(evacuees, target_time=5, use_congestion=False, exit_capacity=1):
    if use_congestion:
        congestion = simulate_exit_congestion(evacuees, exit_capacity)
    else:
        congestion = {
            "congestion_result": {},
            "max_waiting": 0,
            "average_waiting_time": 0,
            "total_congested_people": 0
        }

    escape_times = []

    for evacuee_id in evacuees:
        data = evacuees[evacuee_id]

        if data["success"] and data["escape_time"] is not None:
            escape_times.append(data["escape_time"])

    sorted_times = quick_sort_escape_times(escape_times)
    count_under_target = binary_search_time(sorted_times, target_time)

    total = len(evacuees)
    success_count = len(escape_times)
    fail_count = total - success_count

    avg_time = 0
    if success_count > 0:
        avg_time = sum(escape_times) / success_count

    return {
        "total": total,
        "success_count": success_count,
        "fail_count": fail_count,
        "success_rate": success_count / total * 100 if total > 0 else 0,
        "average_escape_time": avg_time,
        "sorted_escape_times": sorted_times,
        "count_under_target": count_under_target,
        "max_waiting": congestion["max_waiting"],
        "average_waiting_time": congestion["average_waiting_time"],
        "total_congested_people": congestion["total_congested_people"],
        "congestion_result": congestion["congestion_result"]
    }


def print_evacuation_statistics(evacuees, target_time=5, use_congestion=False, exit_capacity=1):
    statistics = get_evacuation_statistics(
        evacuees,
        target_time,
        use_congestion,
        exit_capacity
    )

    print("\n=== 대피 통계 ===")
    print(f"전체 대피자 수: {statistics['total']}")
    print(f"탈출 성공: {statistics['success_count']}")
    print(f"탈출 실패: {statistics['fail_count']}")
    print(f"탈출 성공률: {statistics['success_rate']:.2f}%")
    print(f"탈출 시간 정렬 결과: {statistics['sorted_escape_times']}")
    print(f"t={target_time} 이하 탈출 인원: {statistics['count_under_target']}")
    print(f"평균 탈출 시간: {statistics['average_escape_time']:.2f}")

    if use_congestion:
        print("\n=== 출구 혼잡도 분석 ===")
        print(f"출구 처리 용량: 시간당 {exit_capacity}명")
        print(f"최대 대기 인원: {statistics['max_waiting']}")
        print(f"평균 대기 시간: {statistics['average_waiting_time']:.2f}")
        print(f"대기 발생 인원: {statistics['total_congested_people']}")


def get_simulation_state(evacuees, fire_time, t, target_time=5, use_congestion=False, exit_capacity=1):
    update_all_evacuees_by_time(evacuees, t, fire_time)

    statistics = get_evacuation_statistics(
        evacuees,
        target_time,
        use_congestion,
        exit_capacity
    )

    return {
        "time": t,
        "evacuees": evacuees,
        "statistics": statistics
    }