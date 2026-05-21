from ultralytics import YOLO
import cv2
import numpy as np
import sqlite3
from datetime import datetime

# =========================
# 1. 사용자 설정
# =========================

MODEL_PATH  = "best.pt"
VIDEO_PATH  = "testvideo.mp4"
DB_PATH     = "vehicle_events.db"

START_SEC = 174   # 2분 54초
END_SEC   = 332   # 5분 32초

OUTPUT_VIDEO = "tracked_section.mp4"

# =========================
# 2. Zone polygon 좌표
# =========================

south_zone = [(1136, 1072), (1502, 692), (1616, 792), (1372, 1072)]
north_zone = [(654, 0),    (224, 244),  (132, 110),  (398, 2)]
west_zone  = [(152, 630),  (656, 1072), (400, 1072), (30, 712)]
east_zone  = [(976, 4),    (1420, 336), (1520, 236), (1170, 2)]

ZONES = {
    "south": south_zone,
    "north": north_zone,
    "west":  west_zone,
    "east":  east_zone,
}

# =========================
# 3. 방향 매핑
# =========================

DIRECTION_MAP = {
    ("south", "north"): "남->북",
    ("south", "west"):  "남->서",
    ("south", "east"):  "남->동",
    ("north", "south"): "북->남",
    ("north", "east"):  "북->동",
    ("north", "west"):  "북->서",
    ("east",  "west"):  "동->서",
    ("east",  "north"): "동->북",
    ("east",  "south"): "동->남",
    ("west",  "east"):  "서->동",
    ("west",  "south"): "서->남",
    ("west",  "north"): "서->북",
}

# =========================
# 4. 보조 함수
# =========================

def point_in_polygon(x, y, polygon):
    pts = np.array(polygon, np.int32)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

def get_zone_name(x, y):
    for name, polygon in ZONES.items():
        if point_in_polygon(x, y, polygon):
            return name
    return None

def draw_zones(frame):
    for name, polygon in ZONES.items():
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        lx, ly = polygon[0]
        cv2.putText(frame, name, (lx, max(20, ly - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return frame

def get_vehicle_name(class_id, names_dict):
    return str(names_dict.get(class_id, class_id))

# =========================
# 5. DB 초기화
# =========================

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # 탐지 실행 기록 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detection_runs (
            run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_time    TEXT NOT NULL,
            video_path  TEXT NOT NULL,
            start_sec   REAL NOT NULL,
            end_sec     REAL NOT NULL,
            model_path  TEXT NOT NULL
        )
    """)

    # 차량 이벤트 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL,
            track_id        INTEGER NOT NULL,
            vehicle_type    TEXT,
            entry_zone      TEXT,
            exit_zone       TEXT,
            direction       TEXT,
            entry_time_sec  REAL,
            exit_time_sec   REAL,
            entry_frame     INTEGER,
            exit_frame      INTEGER,
            FOREIGN KEY (run_id) REFERENCES detection_runs(run_id)
        )
    """)

    conn.commit()
    return conn

# =========================
# 6. 초기화
# =========================

conn = init_db(DB_PATH)
cur  = conn.cursor()

# 이번 실행 기록 삽입
cur.execute("""
    INSERT INTO detection_runs (run_time, video_path, start_sec, end_sec, model_path)
    VALUES (?, ?, ?, ?, ?)
""", (datetime.now().isoformat(), VIDEO_PATH, START_SEC, END_SEC, MODEL_PATH))
conn.commit()
run_id = cur.lastrowid
print(f"DB: {DB_PATH}  |  run_id: {run_id}")

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
    conn.close()
    raise SystemExit

fps          = cap.get(cv2.CAP_PROP_FPS)
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_frame = int(START_SEC * fps)
end_frame   = int(END_SEC   * fps)

print(f"fps: {fps}  total_frames: {total_frames}")
print(f"start_frame: {start_frame}  end_frame: {end_frame}")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =========================
# 7. 메인 루프
# =========================

track_memory    = {}
saved_track_ids = set()
current_frame   = start_frame
processed_count = 0
section_total   = end_frame - start_frame

while current_frame < end_frame:
    ret, frame = cap.read()
    if not ret:
        print("\n프레임 읽기 종료")
        break

    processed_count += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.20,
        iou=0.50,
        imgsz=960,
        verbose=False
    )

    result         = results[0]
    annotated_frame = result.plot()
    annotated_frame = draw_zones(annotated_frame)

    if result.boxes is not None and result.boxes.id is not None:
        boxes     = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            zone_name    = get_zone_name(cx, cy)
            vehicle_type = get_vehicle_name(class_id, model.names)

            cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)

            if track_id not in track_memory:
                track_memory[track_id] = {
                    "vehicle_type":  vehicle_type,
                    "entry_zone":    None,
                    "exit_zone":     None,
                    "entry_frame":   None,
                    "exit_frame":    None,
                    "entry_time_sec": None,
                    "exit_time_sec":  None,
                    "visited_zones": []
                }

            info = track_memory[track_id]

            if zone_name is not None:
                if not info["visited_zones"] or info["visited_zones"][-1] != zone_name:
                    info["visited_zones"].append(zone_name)

                if info["entry_zone"] is None:
                    info["entry_zone"]     = zone_name
                    info["entry_frame"]    = current_frame
                    info["entry_time_sec"] = current_frame / fps
                elif zone_name != info["entry_zone"]:
                    info["exit_zone"]     = zone_name
                    info["exit_frame"]    = current_frame
                    info["exit_time_sec"] = current_frame / fps

            # entry/exit 확정 후 DB에 1회만 저장
            if (
                info["entry_zone"] is not None and
                info["exit_zone"]  is not None and
                track_id not in saved_track_ids
            ):
                move_key = (info["entry_zone"], info["exit_zone"])
                if move_key in DIRECTION_MAP:
                    direction = DIRECTION_MAP[move_key]
                    cur.execute("""
                        INSERT INTO vehicle_events
                            (run_id, track_id, vehicle_type,
                             entry_zone, exit_zone, direction,
                             entry_time_sec, exit_time_sec,
                             entry_frame, exit_frame)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        run_id,
                        int(track_id),
                        info["vehicle_type"],
                        info["entry_zone"],
                        info["exit_zone"],
                        direction,
                        round(info["entry_time_sec"], 2),
                        round(info["exit_time_sec"],  2),
                        info["entry_frame"],
                        info["exit_frame"]
                    ))
                    conn.commit()
                    saved_track_ids.add(track_id)
                    print(
                        f"\n저장됨 | ID:{track_id} | {info['vehicle_type']} | "
                        f"{direction} | "
                        f"{info['entry_time_sec']:.2f}s -> {info['exit_time_sec']:.2f}s"
                    )

            cv2.putText(
                annotated_frame,
                f"ID:{track_id}",
                (int(x1), max(20, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

    out.write(annotated_frame)

    print(f"\r처리 중: {processed_count}/{section_total} "
          f"({processed_count / section_total * 100:.2f}%)", end="")

    current_frame += 1

cap.release()
out.release()
conn.close()

print(f"\n완료")
print(f"저장 영상 : {OUTPUT_VIDEO}")
print(f"저장 DB   : {DB_PATH}  (run_id={run_id})")
