# ============================================================
# risk_analyzer.py
# 재난 대피 경로 안내 시스템 - 경로 위험도 분석 모듈
#
# 실행 환경: VSCode / Python 3.9+
# 필요 라이브러리: 없음
#
# 자료구조: 2D 배열(fire_time), 딕셔너리(결과 반환)
# 알고리즘: 위험도 점수 계산 (화재 도달 잔여 시간 기반 가중치)
# ============================================================


def get_fire_distance(pos, fire_time):
    """
    pos 셀의 화재 도달 시각 반환.
    자료구조: 2D 배열(fire_time)
    -1(미도달) 이면 안전 거리 999 반환.
    """
    r, c = pos
    ft = fire_time[r][c]
    return 999 if ft == -1 else ft


def calculate_position_risk(pos, fire_time, current_time):
    """
    단일 셀의 위험도 점수 계산 (0~100).
    알고리즘: 화재 잔여 시간 기반 단계적 가중치

    반환 기준:
      -1 또는 미도달 → 0점(안전)
      remain ≤ 0     → 100점(화재 발생)
      remain ≤ 2     → 80점(임박)
      remain ≤ 5     → 50점(주의)
      remain > 5     → 20점(여유)
    """
    r, c = pos
    # 자료구조: 2D 배열(fire_time) 조회
    fire_arrival_time = fire_time[r][c]

    if fire_arrival_time == -1:
        return 0   # 화재 미도달 → 안전

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
    """
    경로 전체의 평균 위험도 점수 계산.
    자료구조: 리스트(path) 순회, 2D 배열(fire_time) 조회
    """
    if not path:
        return 100

    total_risk = sum(
        calculate_position_risk(pos, fire_time, current_time + step)
        for step, pos in enumerate(path)
    )
    return total_risk / len(path)


def get_risk_level(risk_score):
    """위험도 점수를 텍스트 등급으로 변환."""
    if risk_score >= 70:
        return "위험"
    elif risk_score >= 40:
        return "주의"
    else:
        return "안전"


def compare_exit_safety(exit_paths, fire_time, current_time):
    """
    출구별 경로의 위험도를 비교해 딕셔너리로 반환.
    자료구조: 딕셔너리(결과) — {출구 위치: {risk_score, risk_level}}
    """
    # 자료구조: 딕셔너리(result) — 출구 위치를 키로 위험도 정보 저장
    result = {}
    for exit_pos, path in exit_paths.items():
        risk_score = calculate_path_risk(path, fire_time, current_time)
        result[exit_pos] = {
            "risk_score": risk_score,
            "risk_level": get_risk_level(risk_score),
        }
    return result
