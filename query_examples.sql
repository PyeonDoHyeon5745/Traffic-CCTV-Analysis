-- =============================================
-- vehicle_events.db  SQL 분석 쿼리 예시
-- DB 뷰어: DB Browser for SQLite (무료)
--          또는 python: sqlite3 / pandas
-- =============================================

-- ① 전체 이벤트 조회
SELECT * FROM vehicle_events ORDER BY entry_time_sec;

-- ② 방향별 차량 수
SELECT direction, COUNT(*) AS cnt
FROM vehicle_events
GROUP BY direction
ORDER BY cnt DESC;

-- ③ 차종별 통계
SELECT vehicle_type, COUNT(*) AS cnt
FROM vehicle_events
GROUP BY vehicle_type
ORDER BY cnt DESC;

-- ④ 진입 zone별 통계
SELECT entry_zone, COUNT(*) AS cnt
FROM vehicle_events
GROUP BY entry_zone
ORDER BY cnt DESC;

-- ⑤ 평균 통과 시간 (exit - entry)
SELECT
    direction,
    COUNT(*)                                      AS cnt,
    ROUND(AVG(exit_time_sec - entry_time_sec), 2) AS avg_travel_sec,
    ROUND(MIN(exit_time_sec - entry_time_sec), 2) AS min_sec,
    ROUND(MAX(exit_time_sec - entry_time_sec), 2) AS max_sec
FROM vehicle_events
GROUP BY direction
ORDER BY avg_travel_sec;

-- ⑥ 특정 run_id 결과만 보기
SELECT * FROM vehicle_events WHERE run_id = 1;

-- ⑦ 실행 이력 확인
SELECT * FROM detection_runs;

-- ⑧ 시간대별 차량 수 (30초 구간)
SELECT
    CAST(entry_time_sec / 30 AS INT) * 30 AS time_bucket_sec,
    COUNT(*) AS cnt
FROM vehicle_events
GROUP BY time_bucket_sec
ORDER BY time_bucket_sec;
