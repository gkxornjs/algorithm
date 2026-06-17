# risk_analyzer.py

def get_fire_distance(pos, fire_time):
    r, c = pos
    current_fire_time = fire_time[r][c]

    if current_fire_time == -1:
        return 999

    return current_fire_time


def calculate_position_risk(pos, fire_time, current_time):
    r, c = pos
    fire_arrival_time = fire_time[r][c]

    if fire_arrival_time == -1:
        return 0

    remain_time = fire_arrival_time - current_time

    if remain_time <= 0:
        return 100
    elif remain_time <= 2:
        return 80
    elif remain_time <= 5:
        return 50
    else:
        return 20


def calculate_path_risk(path, fire_time, current_time):
    if not path:
        return 100

    total_risk = 0

    for step, pos in enumerate(path):
        t = current_time + step
        total_risk += calculate_position_risk(pos, fire_time, t)

    return total_risk / len(path)


def get_risk_level(risk_score):
    if risk_score >= 70:
        return "위험"
    elif risk_score >= 40:
        return "주의"
    else:
        return "안전"


def compare_exit_safety(exit_paths, fire_time, current_time):
    result = {}

    for exit_pos, path in exit_paths.items():
        risk_score = calculate_path_risk(path, fire_time, current_time)
        result[exit_pos] = {
            "risk_score": risk_score,
            "risk_level": get_risk_level(risk_score)
        }

    return result