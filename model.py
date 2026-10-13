from lift_counter import LiftCounter
import math
from collections import deque
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def open_video(path):
    # เปิดวิดีโอ คืน (cap, ข้อมูลวิดีโอ)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ ไม่สามารถเปิดไฟล์วิดีโอได้ กรุณาตรวจสอบว่ามีไฟล์อยู่จริงที่พาธ: {path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    if info["fps"] <= 0 or info["frames"] <= 0:
        raise ValueError(f"❌ วิดีโอเสียหายหรือไม่สามารถอ่าน Metadata ได้: {info}")
    return cap, info

def read_frame(path, frame_number):
    # อ่านเฟรมที่ต้องการ (นับจาก 1)
    cap, _ = open_video(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"❌ ไม่สามารถอ่านเฟรมที่ {frame_number} ได้")
    return frame

def show_images(images, titles, width=4):
    # แสดงภาพ BGR หลายภาพเรียงกัน
    fig, axes = plt.subplots(1, len(images), figsize=(width * len(images), width * 1.6))
    axes = np.atleast_1d(axes)
    for ax, img, title in zip(axes, images, titles):
        if img is not None and img.size > 0:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))     # BGR → RGB ก่อนแสดง
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def create_pose_detector():
    # สร้างตัวตรวจจับท่าทาง (สร้างครั้งเดียว ใช้ได้ทั้งคลิป)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1)
    return mp_vision.PoseLandmarker.create_from_options(options)

def extract_keypoints(detector, frame, frame_index):
    # ② คืน xy (33, 2) พิกเซล · world (33, 3) เมตร · conf (33,)   (เป็น NaN ถ้าไม่พบคน)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(frame_index * 1000 / FPS)
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

    xy = np.full((33, 2), np.nan)
    world = np.full((33, 3), np.nan)
    conf = np.full(33, np.nan)
    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks[0]):
            xy[i] = [lm.x * w, lm.y * h]
            conf[i] = min(lm.visibility, lm.presence)
        for i, lm in enumerate(result.pose_world_landmarks[0]):
            world[i] = [lm.x, lm.y, lm.z]
    return xy, world, conf

MIN_CONFIDENCE     = 0.5

MAX_PREDICT_FRAMES = 5

HISTORY_FRAMES     = 3

WORLD_WINDOW       = 5

def create_kalman(x, y):
    # Kalman filter แบบความเร็วคงที่  สถานะ = [x, y, vx, vy]  ค่าวัด = [x, y]
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    acc = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], np.float32)
    kf.processNoiseCov = 16.0 * acc @ acc.T                     # การเคลื่อนที่ไม่แน่นอนแค่ไหน
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 25.0  # ค่าวัดคลาดเคลื่อนประมาณ 5 px
    kf.errorCovPost = np.diag([25, 25, 400, 400]).astype(np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf

class PointTracker:
    # ติดตาม + ลดการสั่นของ keypoint 1 จุด

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.gate = max(25.0, 0.04 * math.hypot(width, height))   # ระยะกระโดดสูงสุดที่ยอมรับ (px)
        self.kalman = None       # None = ยังไม่ได้เริ่มติดตาม
        self.missing = 0         # จำนวนเฟรมติดกันที่ไม่มีค่าวัดที่ดี
        self.history = []        # ตำแหน่งล่าสุด สำหรับ moving average

    def update(self, x, y, conf):
        # คืน (ตำแหน่งที่กรองแล้ว, วัดได้จริงในเฟรมนี้หรือไม่)
        good = conf >= MIN_CONFIDENCE and 0 <= x < self.width and 0 <= y < self.height   # NaN → False
        point, observed = None, False

        if self.kalman is None:                          # ยังไม่ได้ติดตาม → เริ่มเมื่อค่าวัดดี
            if good:
                self.kalman = create_kalman(x, y)
                point, observed = np.array([x, y]), True
        else:
            predicted = self.kalman.predict()[:2, 0]     # (ก-1) predict
            jump = math.hypot(x - predicted[0], y - predicted[1]) if good else 0
            if good and jump <= self.gate:               # (ก-2) correct ด้วยค่าวัด
                self.kalman.correct(np.array([[x], [y]], np.float32))
                point, observed = np.array([x, y]), True
                self.missing = 0
            else:                                        # ไม่มีค่าวัดที่ดี → ใช้ค่าทาย
                self.missing += 1
                inside = 0 <= predicted[0] < self.width and 0 <= predicted[1] < self.height
                if self.missing <= MAX_PREDICT_FRAMES and inside:
                    point = predicted
                else:                                    # หายนานเกินไป → เริ่มใหม่
                    self.kalman, self.missing, self.history = None, 0, []

        if point is None:
            return np.array([np.nan, np.nan]), False
        self.history = (self.history + [point])[-(HISTORY_FRAMES + 1):]   # (ข) moving average
        return np.mean(self.history, axis=0), observed

class PoseSmoother:
    # PointTracker 33 ตัว (สำหรับภาพ) + ค่าเฉลี่ยย้อนหลังของโครงร่าง 3 มิติ

    def __init__(self, width, height):
        self.trackers = [PointTracker(width, height) for _ in range(33)]
        self.world_history = deque(maxlen=WORLD_WINDOW)

    def update(self, xy, world, conf):
        xy_s = np.full((33, 2), np.nan)
        observed = np.zeros(33, dtype=bool)
        for i in range(33):
            xy_s[i], observed[i] = self.trackers[i].update(xy[i, 0], xy[i, 1], conf[i])
        self.world_history.append(world)
        world_s = np.nanmean(np.array(self.world_history), axis=0)    # (ค)
        return xy_s, world_s, observed

SKELETON = [(11, 12), (11, 23), (12, 24), (23, 24),                  # ลำตัว
            (11, 13), (13, 15), (15, 17), (15, 19),                  # แขนซ้าย
            (12, 14), (14, 16), (16, 18), (16, 20),                  # แขนขวา
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),        # ขาซ้าย
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),        # ขาขวา
            (0, 11), (0, 12)]

def pt(p):
    # แปลงพิกัดทศนิยมเป็นจุดจำนวนเต็มสำหรับ OpenCV
    return (int(round(p[0])), int(round(p[1])))

def ui_scale(img):
    # ขนาดตัวอักษร/เส้น เทียบกับภาพกว้าง 720 px
    return img.shape[1] / 720

def draw_skeleton(img, xy_s, observed):
    # ④ วาดโครงกระดูกลงบนภาพ
    s = ui_scale(img)
    for a, b in SKELETON:
        if np.isfinite(xy_s[a]).all() and np.isfinite(xy_s[b]).all():
            cv2.line(img, pt(xy_s[a]), pt(xy_s[b]), (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    for a, b in SKELETON:
        for p in (a, b):
            if np.isfinite(xy_s[p]).all():
                color = (0, 200, 0) if observed[p] else (0, 140, 255)
                cv2.circle(img, pt(xy_s[p]), max(2, int(4 * s)), color, -1, cv2.LINE_AA)
    return img

NOSE_ABOVE_FOOT = 0.905

CALIB_SECONDS   = 1.5

CAMERA_FOV_DEG  = 70.0

WAIST_FRACTION  = 0.30

FEET = [29, 30, 31, 32]

LOWER = [27, 28, 29, 30, 31, 32]

PNP_POINTS = [0, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

def unit(v):
    # ทำเวกเตอร์ให้ยาว 1
    return v / np.linalg.norm(v)

def mean_point(points, ids):
    # จุดเฉลี่ยของหลาย landmark (ข้ามค่า NaN)
    return np.nanmean(points[ids], axis=0)

def calc_cm_per_px(xy_frames, height_cm):
    # ซม. ต่อพิกเซล จากระยะพื้น→จมูกในภาพ
    distances = []
    for xy in xy_frames:
        floor_y = np.nanmax(xy[FEET, 1])
        distances.append(floor_y - xy[0, 1])
    return height_cm * NOSE_ABOVE_FOOT / np.nanmedian(distances)

def camera_matrix(width, height):
    # เมทริกซ์กล้องรูเข็ม K จาก FOV
    f = (max(width, height) / 2) / math.tan(math.radians(CAMERA_FOV_DEG) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])

def calc_world_scale(world_frames, height_cm):
    # ตัวคูณที่ทำให้โครงร่าง 3 มิติสูง (พื้น→จมูก) เท่าผู้ยกจริง
    heights = []
    for world in world_frames:
        axis = unit(mean_point(world, [23, 24]) - mean_point(world, LOWER))  # เท้า→สะโพก
        heights.append(np.nanmax((world[0] - world[FEET]) @ axis) * 100)    # พื้น→จมูก (ซม.)
    return height_cm * NOSE_ABOVE_FOOT / np.nanmedian(heights)

def solve_camera_pose(P, xy, observed, K, previous=None):
    # หา R, t ที่ทำให้จุด 3 มิติ P ฉายลงภาพแล้วตรงกับ xy มากที่สุด
    ids = [i for i in PNP_POINTS if observed[i]]
    if len(ids) < 6:
        return None
    if previous is None:
        ok, rvec, tvec = cv2.solvePnP(P[ids], xy[ids], K, None)
    else:                                                    # ใช้ผลเฟรมก่อนเป็นจุดเริ่ม → นิ่งและเร็ว
        ok, rvec, tvec = cv2.solvePnP(P[ids], xy[ids], K, None,
                                      previous["rvec"].copy(), previous["tvec"].copy(), True)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return {"R": R, "t": tvec.ravel(), "rvec": rvec, "tvec": tvec}

def to_camera_3d(xy, P, pose, K):
    # ตำแหน่งในภาพ (u, v) + ความลึก Z จากโครงร่าง → พิกัดกล้อง 3 มิติ (33, 3) ซม.
    P_cam = P @ pose["R"].T + pose["t"]                      # โครงร่างในพิกัดกล้อง
    Z = P_cam[:, 2]
    X = (xy[:, 0] - K[0, 2]) * Z / K[0, 0]
    Y = (xy[:, 1] - K[1, 2]) * Z / K[1, 1]
    C = np.column_stack([X, Y, Z])
    missing = np.isnan(xy[:, 0])
    C[missing] = P_cam[missing]                              # จุดที่ไม่เห็นในภาพ → ใช้โครงร่างแทน
    return C

def project(points, K):
    # จุดในพิกัดกล้อง (…, 3) → ตำแหน่งในภาพ (…, 2) พิกเซล
    points = np.asarray(points, dtype=float)
    u = K[0, 0] * points[..., 0] / points[..., 2] + K[0, 2]
    v = K[1, 1] * points[..., 1] / points[..., 2] + K[1, 2]
    return np.stack([u, v], axis=-1)

def body_frame(C, up, floor=None):
    # กรอบพิกัดร่างกาย: O, up, side, front  (floor = จุดบนพื้นจาก calibrate)
    ankle = mean_point(C, [27, 28])
    side = C[24] - C[23]
    side = unit(side - (side @ up) * up)                     # เส้นสะโพกในแนวราบ
    front = unit(np.cross(side, up))                         # ตั้งฉากกับระนาบสะโพก
    toe, heel = mean_point(C, [31, 32]), mean_point(C, [29, 30])
    if (toe - heel) @ front < 0:                             # ให้ front ชี้ไปทางปลายเท้า
        front = -front
    if floor is None:
        drop = np.nanmin((C[FEET] - ankle) @ up)             # พื้น = จุดเท้าที่ต่ำที่สุด
    else:
        drop = (floor - ankle) @ up
    O = ankle + up * drop                                    # กึ่งกลางข้อเท้า ที่ระดับพื้น
    if not (np.isfinite(O).all() and np.isfinite(front).all()):
        return None
    return {"O": O, "up": up, "side": side, "front": front}

def body_coords(p, body):
    # จุด 3 มิติ → (ด้านหน้า, ความสูง, ด้านข้าง) ซม. เทียบกับ O
    d = p - body["O"]
    return d @ body["front"], d @ body["up"], d @ body["side"]

def twist_angle(world):
    # มุม (มีเครื่องหมาย) ระหว่างเส้นไหล่กับเส้นสะโพก รอบแกนลำตัว — ใช้ในขั้นที่ ⑦
    axis = unit(mean_point(world, [23, 24]) - mean_point(world, LOWER))
    hip = world[24] - world[23]
    shoulder = world[12] - world[11]
    hip = unit(hip - (hip @ axis) * axis)
    shoulder = unit(shoulder - (shoulder @ axis) * axis)
    return math.degrees(math.atan2(np.cross(hip, shoulder) @ axis, hip @ shoulder))

def calibrate(samples, height_cm, width, height):
    # ⑤ samples = รายการ {"xy", "world", "observed"} ของช่วงยืนนิ่ง → dict calib
    calib = {"height_cm": height_cm}
    calib["cm_per_px"] = calc_cm_per_px([s["xy"] for s in samples], height_cm)               # 5.1
    calib["K"] = camera_matrix(width, height)
    calib["camera_dist_cm"] = calib["K"][0, 0] * calib["cm_per_px"]
    calib["world_scale"] = calc_world_scale([s["world"] for s in samples], height_cm)          # 5.2

    C_list, axes, pose = [], [], None                                                        # พิกัด 3 มิติของแต่ละเฟรม
    for s in samples:
        P = s["world"] * 100 * calib["world_scale"]
        pose = solve_camera_pose(P, s["xy"], s["observed"], calib["K"], pose)
        if pose is None:
            continue
        C = to_camera_3d(s["xy"], P, pose, calib["K"])
        C_list.append(C)
        axes.append(unit(mean_point(C, [23, 24]) - mean_point(C, LOWER)))
    if not axes or not np.isfinite(calib["cm_per_px"]) or calib["cm_per_px"] <= 0:
        raise ValueError("ปรับเทียบไม่ได้ กรุณาให้เห็นทั้งตัวและยืนนิ่ง 1.5 วินาทีแรก")
    calib["up"] = unit(np.mean(axes, axis=0))                                                # 5.3 แนวดิ่ง
    floors = []                                                                              # จุดบนพื้น
    for C in C_list:
        body = body_frame(C, calib["up"])
        if body is not None:
            floors.append(body["O"])
    calib["floor"] = np.median(floors, axis=0)

    knee, hip, shoulder = [], [], []                                                         # ระดับเข่า/สะโพก/ไหล่
    for C in C_list:
        body = body_frame(C, calib["up"], calib["floor"])
        if body is None:
            continue
        knee.append(body_coords(mean_point(C, [25, 26]), body)[1])
        hip.append(body_coords(mean_point(C, [23, 24]), body)[1])
        shoulder.append(body_coords(mean_point(C, [11, 12]), body)[1])
    calib["knee_cm"] = np.median(knee)
    calib["shoulder_cm"] = np.median(shoulder)
    calib["waist_cm"] = np.median(hip) + WAIST_FRACTION * (calib["shoulder_cm"] - np.median(hip))
    calib["twist_baseline"] = np.nanmedian([twist_angle(s["world"]) for s in samples])       # ใช้ในขั้นที่ ⑦
    return calib

def draw_body_frame(img, C, calib):
    # วาด O · แนวดิ่ง · ลูกศร front · ระดับเข่า/เอว/ไหล่
    body = body_frame(C, calib["up"], calib["floor"])
    K, s = calib["K"], ui_scale(img)
    O, up, side, front = body["O"], body["up"], body["side"], body["front"]
    cv2.line(img, pt(project(O, K)), pt(project(O + up * calib["shoulder_cm"], K)), (255, 255, 0), 2)
    cv2.arrowedLine(img, pt(project(O, K)), pt(project(O + front * 40, K)), (255, 0, 255), 3, tipLength=0.25)
    cv2.circle(img, pt(project(O, K)), int(7 * s), (255, 0, 255), -1)
    cv2.putText(img, "O", pt(project(O, K) + 10 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 0, 255), 2)
    for name, color in [("knee", (0, 215, 255)), ("waist", (0, 165, 255)), ("shoulder", (60, 60, 255))]:
        level = O + up * calib[name + "_cm"]
        a, b = project(level - side * 25, K), project(level + side * 25, K)
        cv2.line(img, pt(a), pt(b), color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{name} {calib[name + '_cm']:.0f} cm", pt((max(a[0], b[0]) + 5, a[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, color, 2)
    return img

def orientation_angle(C):
    # ⑥ มุมระหว่างเส้นสะโพกกับระนาบกล้อง (0 = หันเข้ากล้อง, 90 = หันข้าง)
    d = C[24] - C[23]
    yaw = abs(math.degrees(math.atan2(d[2], d[0])))    # 0…180
    return min(yaw, 180 - yaw)

TWIST_WARN_DEG = 45.0

def twist_from_baseline(world, calib):
    # ⑦ มุมบิดตัว (องศา, ค่าบวก) เทียบกับท่ายืนตรง
    diff = twist_angle(world) - calib["twist_baseline"]
    diff = (diff + 180) % 360 - 180                    # ให้อยู่ในช่วง −180…180
    return abs(diff)

REACH_LIMITS_CM = [7 * 2.54, 12 * 2.54]

ROW_NAMES = ["Above Shoulder", "Waist-Shoulder", "Knee-Waist", "Below Knee"]

COL_NAMES = ["Near", "Mild", "Extended"]

WEIGHT_LIMITS_KG = [[29, 18, 14],      # Above Shoulder
                    [32, 23, 18],      # Waist to Shoulder
                    [41, 25, 18],      # Knee to Waist  (สูงสุด)
                    [31, 23, 16]]

LEFT_HAND, RIGHT_HAND = [15, 17, 19], [16, 18, 20]

def table_row(v_cm, calib):
    # แถวของตารางจากความสูงมือ
    if v_cm >= calib["shoulder_cm"]:
        return 0
    if v_cm >= calib["waist_cm"]:
        return 1
    if v_cm >= calib["knee_cm"]:
        return 2
    return 3

def table_col(h_cm):
    # คอลัมน์ของตารางจากระยะ H
    if h_cm < REACH_LIMITS_CM[0]:
        return 0
    if h_cm < REACH_LIMITS_CM[1]:
        return 1
    return 2

def measure_hands(C, xy, calib):
    # ⑧-1 คืน H, V, ช่องในตาราง และน้ำหนักสูงสุด (None ถ้าวัดไม่ได้)
    body = body_frame(C, calib["up"], calib["floor"])
    if body is None:
        return None
    hands_3d = np.nanmean([mean_point(C, LEFT_HAND), mean_point(C, RIGHT_HAND)], axis=0)
    hands_px = np.nanmean([mean_point(xy, LEFT_HAND), mean_point(xy, RIGHT_HAND)], axis=0)
    if np.isnan(hands_3d).any():
        return None
    if np.isnan(hands_px).any():                            # มือหลุดจากภาพ → ฉายจาก 3 มิติแทน
        hands_px = project(hands_3d, calib["K"])

    front, v_cm, side = body_coords(hands_3d, body)
    h_cm = math.hypot(front, side)                          # NIOSH H
    row, col = table_row(v_cm, calib), table_col(h_cm)
    return {"H": h_cm, "V": v_cm, "front": front, "side": side,
            "row": row, "col": col, "limit_kg": WEIGHT_LIMITS_KG[row][col],
            "hands_px": hands_px, "body": body,
            "direction": -1 if front < -REACH_LIMITS_CM[0] else 1}

TABLE_LENGTH_CM = 55.0

ABOVE_MARGIN_CM = 30.0

HEADER_CM       = 10.0

PX_PER_CM       = 8

CELL_ALPHA      = 0.45

def cell_color(kg):
    # สีเดียวไล่ความเข้ม: น้ำหนักมาก (ปลอดภัยกว่า) = ครีมอ่อน · น้อย = แดงเข้ม
    t = (41 - kg) / (41 - 14)
    light, dark = np.array([160, 225, 255]), np.array([20, 20, 170])
    return tuple(int(c) for c in light + (dark - light) * t)

def make_table_image(calib, highlight, axis_on_left):
    # ⑧-2 วาดตารางแบน: คืน (ภาพสี, ภาพความทึบ 0–1)
    top_cm = calib["shoulder_cm"] + ABOVE_MARGIN_CM + HEADER_CM
    w, h = int(TABLE_LENGTH_CM * PX_PER_CM), int(top_cm * PX_PER_CM)
    color = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)  # เปลี่ยนเป็น uint8 เพื่อให้ putText วาดได้

    col_edges = [0, REACH_LIMITS_CM[0], REACH_LIMITS_CM[1], TABLE_LENGTH_CM]            # ซม. จากลำตัว
    row_edges = [calib["shoulder_cm"] + ABOVE_MARGIN_CM, calib["shoulder_cm"],
                 calib["waist_cm"], calib["knee_cm"], 0]                               # ซม. จากพื้น (บน→ล่าง)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def to_x(cm):
        x = int(cm * PX_PER_CM)
        return x if axis_on_left else w - x

    def to_y(cm):
        return int((top_cm - cm) * PX_PER_CM)

    def write(text, pos, size, thick, rgb=(255, 255, 255)):
        cv2.putText(color, text, pos, font, size, rgb, thick, cv2.LINE_AA)
        cv2.putText(alpha, text, pos, font, size, 255, thick, cv2.LINE_AA)

    for r in range(4):
        for c in range(3):
            x0, x1 = sorted([to_x(col_edges[c]), to_x(col_edges[c + 1])])
            y0, y1 = to_y(row_edges[r]), to_y(row_edges[r + 1])
            kg = WEIGHT_LIMITS_KG[r][c]
            is_hand_cell = (r, c) == highlight

            cv2.rectangle(color, (x0, y0), (x1, y1), (0, 230, 255) if is_hand_cell else cell_color(kg), -1)

            # เก็บค่าความโปร่งแสงลงในอาเรย์ uint8 (0-255)
            alpha_val = int(255 * (0.85 if is_hand_cell else CELL_ALPHA))
            alpha[y0:y1, x0:x1] = alpha_val

            cv2.rectangle(color, (x0, y0), (x1, y1), (255, 255, 255), 2)
            cv2.rectangle(alpha, (x0, y0), (x1, y1), 255, 2)
            write(str(kg), ((x0 + x1) // 2 - 22, (y0 + y1) // 2 + 15), 1.6, 4,
                  (0, 0, 0) if is_hand_cell else (255, 255, 255))
        x_label = 8 if not axis_on_left else w - 190
        write(ROW_NAMES[r], (x_label, to_y(row_edges[r]) + 26), 0.7, 2)
    for c in range(3):
        x0, x1 = sorted([to_x(col_edges[c]), to_x(col_edges[c + 1])])
        write(COL_NAMES[c], (x0 + 8, to_y(row_edges[0]) - 14), 0.7, 2)
    write("MAX WEIGHT (kg)", (w // 2 - 120, 30), 0.8, 2)

    # แปลงอาเรย์ความโปร่งแสงกลับเป็น float32 (0.0-1.0) เพื่อใช้งาน
    return color, alpha.astype(np.float32) / 255.0

def draw_weight_table(img, calib, m):
    # ⑧-3 ฉายมุมตาราง 3 มิติลงภาพ แล้ว warpPerspective ภาพตารางลงไป
    body, K = m["body"], calib["K"]
    O, up, front = body["O"], body["up"], body["front"] * m["direction"]
    top_cm = calib["shoulder_cm"] + ABOVE_MARGIN_CM + HEADER_CM
    corners_3d = np.array([O,                                               # ล่าง-ลำตัว
                           O + front * TABLE_LENGTH_CM,                     # ล่าง-ปลาย
                           O + front * TABLE_LENGTH_CM + up * top_cm,       # บน-ปลาย
                           O + up * top_cm])                                # บน-ลำตัว
    if (corners_3d[:, 2] < 10).any():                                       # อยู่หลังกล้อง → ไม่วาด
        return img
    corners_px = project(corners_3d, K).astype(np.float32)
    if abs(cv2.contourArea(corners_px)) < 0.002 * img.shape[0] * img.shape[1]:
        return img                                                          # แผ่นหันสันเข้ากล้อง → ไม่วาด

    axis_on_left = corners_px[1, 0] >= corners_px[0, 0]                     # ตารางยื่นไปทางขวาของภาพ?
    table, alpha = make_table_image(calib, (m["row"], m["col"]), axis_on_left)
    th, tw = alpha.shape
    x_body, x_end = (0, tw) if axis_on_left else (tw, 0)
    table_corners = np.float32([[x_body, th], [x_end, th], [x_end, 0], [x_body, 0]])

    M = cv2.getPerspectiveTransform(table_corners, corners_px)              # เมทริกซ์ 3×3
    size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(table, M, size)
    warped_alpha = cv2.warpPerspective(alpha, M, size)[:, :, None]
    img[:] = (img * (1 - warped_alpha) + warped * warped_alpha).astype(np.uint8)
    return img

def draw_hand_label(img, calib, m):
    # ⑧-4 จุด (H, V) บนตาราง + ป้ายน้ำหนักสูงสุด
    body, K, s = m["body"], calib["K"], ui_scale(img)
    O, up, front = body["O"], body["up"], body["front"] * m["direction"]
    on_table = project(O + front * m["H"] + up * m["V"], K)                 # (H, V) บนแผ่นตาราง
    hands = m["hands_px"]
    cv2.line(img, pt(hands), pt(on_table), (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.circle(img, pt(on_table), int(7 * s), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt(hands), int(16 * s), (0, 230, 255), max(2, int(3 * s)), cv2.LINE_AA)

    # ป้ายวางที่ขอบปลายตาราง ระดับเดียวกับมือ
    anchor = project(O + front * (TABLE_LENGTH_CM + 3) + up * m["V"], K)
    box_w, box_h = int(230 * s), int(80 * s)
    x0 = anchor[0] if anchor[0] >= project(O, K)[0] else anchor[0] - box_w
    x0 = int(np.clip(x0, 5, img.shape[1] - box_w - 5))
    y0 = int(np.clip(anchor[1] - box_h / 2, 5, img.shape[0] - box_h - 5))
    cv2.line(img, pt(on_table), (x0 + box_w // 2, y0 + box_h // 2), (0, 230, 255), max(2, int(3 * s)))
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 230, 255), -1)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"MAX {m['limit_kg']} kg", (x0 + int(10 * s), y0 + int(35 * s)), font, 1.0 * s,
                (0, 0, 0), max(2, int(3 * s)), cv2.LINE_AA)
    cv2.putText(img, f"{ROW_NAMES[m['row']]} | {COL_NAMES[m['col']]}", (x0 + int(10 * s), y0 + int(55 * s)),
                font, 0.45 * s, (0, 0, 0), max(1, int(1.5 * s)), cv2.LINE_AA)
    cv2.putText(img, f"H {m['H']:.0f} cm   V {m['V']:.0f} cm", (x0 + int(10 * s), y0 + int(72 * s)),
                font, 0.45 * s, (0, 0, 0), max(1, int(1.5 * s)), cv2.LINE_AA)
    return img

def draw_info_panel(img, lines):
    # แผงข้อความมุมซ้ายบน: lines = [(ข้อความ, สี), ...]
    s = ui_scale(img)
    line_h = int(26 * s)
    overlay = img.copy()
    # เพิ่มความกว้างเป็น 420 * s เพื่อให้ไม่โดนตัดขอบเมื่อตัวอักษรยาวขึ้น
    cv2.rectangle(overlay, (0, 0), (int(420 * s), int(15 * s) + line_h * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (int(15 * s), int(28 * s) + i * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * s, color, max(1, int(1.5 * s)), cv2.LINE_AA)
    return img

def run_pipeline(video_path, out_path, height_cm, max_frames=None, v_threshold=20.0, progress=None):
    cap, info = open_video(video_path)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"],
                             (info["width"], info["height"]))
    smoother = PoseSmoother(info["width"], info["height"])
    if not writer.isOpened():
        cap.release()
        raise ValueError("ไม่สามารถสร้างวิดีโอผลลัพธ์ได้")
    n_calib = int(round(CALIB_SECONDS * info["fps"]))
    calib, samples, pose = None, [], None
    records = []
    WHITE, YELLOW, RED = (255, 255, 255), (0, 230, 255), (80, 80, 255)
    ORANGE = (0, 165, 255)

    # ตัวแปรสำหรับการนับรอบ Real-time ด้วยสัญญาณแบบมีบัฟเฟอร์สัญญาณ V_cm
    counter = LiftCounter(info["fps"], v_threshold)
    v_history = []
    lift_count = 0
    trough_val = None
    look_for_peak = False

    try:
        with create_pose_detector() as detector:
            i = 0
            while max_frames is None or i < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                xy, world, conf = extract_keypoints(detector, frame, i)
                xy_s, world_s, observed = smoother.update(xy, world, conf)
                img = frame.copy()
                row = {"frame_number": i + 1, "time_s": (i + 1) / info["fps"]}

                if calib is None:
                    samples.append({"xy": xy_s, "world": world_s, "observed": observed})
                    if len(samples) == n_calib:
                        calib = calibrate(samples, height_cm, info["width"], info["height"])
                    img = draw_skeleton(img, xy_s, observed)
                    img = draw_info_panel(img, [(f"CALIBRATING... stand still {len(samples)}/{n_calib}", YELLOW)])
                else:
                    P = world_s * 100 * calib["world_scale"]
                    pose = solve_camera_pose(P, xy_s, observed, calib["K"], pose)
                    m = None
                    if pose is not None:
                        C = to_camera_3d(xy_s, P, pose, calib["K"])
                        row["orientation_deg"] = orientation_angle(C)
                        m = measure_hands(C, xy_s, calib)
                    row["twist_deg"] = twist_from_baseline(world_s, calib)

                    if m is not None:
                        lift_count = counter.update(m["V"], i)

                        img = draw_weight_table(img, calib, m)
                    else:
                        lift_count = counter.update(float("nan"), i)

                    img = draw_skeleton(img, xy_s, observed)
                    lines = [
                        (f"scale   {calib['cm_per_px']:.3f} cm/px", WHITE),
                        (f"orient  {row.get('orientation_deg', np.nan):.0f} deg to camera", WHITE),
                        (f"twist   {row['twist_deg']:.0f} deg", RED if row["twist_deg"] > TWIST_WARN_DEG else WHITE),
                        (f"LIFT COUNT: {lift_count}", ORANGE)
                    ]
                    if m is not None:
                        img = draw_hand_label(img, calib, m)
                        lines += [
                            (f"hands   H {m['H']:.0f} cm | V {m['V']:.0f} cm", WHITE),
                            (f"zone    {ROW_NAMES[m['row']]} | {COL_NAMES[m['col']]}", WHITE),
                            (f"LIMIT   {m['limit_kg']} kg", YELLOW)
                        ]
                        row.update({"H_cm": m["H"], "V_cm": m["V"], "zone": ROW_NAMES[m["row"]],
                                    "reach_class": COL_NAMES[m["col"]], "limit_kg": m["limit_kg"]})
                    img = draw_info_panel(img, lines)

                writer.write(img)
                row["lift_count_realtime"] = lift_count
                records.append(row)
                i += 1
                if progress and (i % 10 == 0 or i == info["frames"]):
                    progress(i, info["frames"])
                if i % 50 == 0:
                    print(f"  ประมวลผลแล้ว {i} เฟรม | Lifts: {lift_count}")
    finally:
        cap.release()
        writer.release()
    print(f"เสร็จเรียบร้อย {i} เฟรม (นับการยกได้ทั้งหมด {lift_count} ครั้ง)")
    return pd.DataFrame(records), calib

def count_lifts_with_rest(results, calib, v_threshold=20.0, rest_tol=10.0):
    """
    ฟังก์ชันนับจำนวนครั้งการยกโดยคำนวณจากค่า V (ความสูงของมือ)
    - มีการเปลี่ยนแปลงของ V สูงขึ้นหรือลดลงจากจุดเริ่มอย่างน้อย v_threshold (20 ซม.)
    - หลังจากนับแล้ว จะล็อคไว้ไม่นับซ้ำจนกว่าผู้ยกจะกลับเข้าสู่สถานะยืนพัก (Resting state)
    - สถานะยืนพัก: มือกลับมาอยู่ในระดับใกล้เคียงกับช่วง Calibrate (ยืนนิ่งต้นคลิป) ภายในช่วงความต่างไม่เกิน rest_tol (ซม.)
    """
    # หาค่าเฉลี่ยระดับมือช่วงยืนนิ่งเริ่มต้น (Calibrate)
    calib_frames = int(round(CALIB_SECONDS * FPS))
    v_calib_mean = results.iloc[:calib_frames]["V_cm"].mean()

    lift_count = 0
    state = "REST" # สถานะเริ่มต้น: REST, LIFTING, COUNTED
    start_v = v_calib_mean
    lift_log = []

    for idx, row in results.iterrows():
        v = row["V_cm"]
        if np.isnan(v):
            continue

        frame_num = row["frame_number"]
        time_s = row["time_s"]

        if state == "REST":
            # ตรวจสอบว่าเริ่มมีการเคลื่อนไหวของมือออกจากระดับพักมากกว่าเกณฑ์หรือไม่
            if abs(v - v_calib_mean) > 5.0: # เริ่มขยับ
                state = "LIFTING"
                start_v = v

        elif state == "LIFTING":
            # ดูว่าระยะยกขึ้นหรือลงแตกต่างจากจุดเริ่มเคลื่อนไหวเกิน v_threshold ซม.
            if abs(v - start_v) >= v_threshold:
                lift_count += 1
                state = "COUNTED"
                lift_log.append({"frame": frame_num, "time": time_s, "V": v, "type": "Upward" if v > start_v else "Downward"})
                print(f"[LIFT {lift_count}] ตรวจพบการยกที่เฟรม {frame_num:.0f} (เวลา {time_s:.2f} วินาที) | V = {v:.1f} ซม.")

        elif state == "COUNTED":
            # ล็อคไว้จนกว่าระดับมือจะกลับเข้าใกล้ระดับพักเริ่มต้น (Resting state)
            if abs(v - v_calib_mean) <= rest_tol:
                state = "REST"

    return lift_count, lift_log

def count_lifts_diagnostics(results, calib, v_threshold=20.0, rest_tol=10.0):
    calib_frames = int(round(CALIB_SECONDS * FPS))
    v_calib_mean = results.iloc[:calib_frames]["V_cm"].mean()

    states = []
    lift_count = 0
    state = "REST" # REST, LIFTING, COUNTED
    start_v = v_calib_mean
    lift_log = []

    for idx, row in results.iterrows():
        v = row["V_cm"]
        if np.isnan(v):
            states.append(state)
            continue

        frame_num = row["frame_number"]
        time_s = row["time_s"]

        if state == "REST":
            if abs(v - v_calib_mean) > 5.0:
                state = "LIFTING"
                start_v = v

        elif state == "LIFTING":
            if abs(v - start_v) >= v_threshold:
                lift_count += 1
                state = "COUNTED"
                lift_log.append({"frame": frame_num, "time": time_s, "V": v})
                print(f"[LIFT {lift_count}] ตรวจพบที่เฟรม {frame_num:.0f} (เวลา {time_s:.2f} วินาที) | V = {v:.1f} ซม.")

        elif state == "COUNTED":
            # ตรวจสอบการกลับสู่พื้นที่พัก
            if abs(v - v_calib_mean) <= rest_tol:
                state = "REST"

        states.append(state)

    # วาดกราฟแสดงผลค่า V ควบคู่ไปกับเส้นระดับเฉลี่ยและขอบเขตสถานะ
    plt.figure(figsize=(14, 5))
    plt.plot(results["time_s"], results["V_cm"], label="Hand Height (V_cm)", color="#1a73e8", lw=2.5)
    plt.axhline(v_calib_mean, color="green", linestyle="--", label=f"Calibrated Rest ({v_calib_mean:.1f} cm)")
    plt.axhspan(v_calib_mean - rest_tol, v_calib_mean + rest_tol, color="green", alpha=0.15, label="Resting Zone")

    # แสดงจุดที่นับรอบสำเร็จ
    for log in lift_log:
        plt.plot(log["time"], log["V"], "ro", markersize=10, label="Count Trigger")

    plt.title("วิเคราะห์การนับจำนวนยกและช่วงพักของมือ")
    plt.xlabel("Time (s)")
    plt.ylabel("V (cm)")
    plt.grid(True, alpha=0.25)
    # ล้าง Label ซ้ำ
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

    return lift_count, states

def count_lifts_robust(results, calib, v_threshold=20.0):
    """
    ฟังก์ชันนับรอบการยกโดยใช้หลักการตรวจจับจุดวกกลับต่ำสุด-สูงสุด (Trough-to-Peak)
    - หาจุดต่ำสุดตอนก้มหยิบ (Trough) และจุดสูงสุดตอนยกขึ้น (Peak)
    - ถ้าระยะห่างของจุดต่ำสุดและสูงสุดต่างกันเกิน v_threshold จะนับเป็นการยก 1 รอบอย่างแม่นยำ
    """
    v_values = results["V_cm"].values
    time_values = results["time_s"].values
    frames = results["frame_number"].values

    # ขจัดค่า NaN ชั่วคราวสำหรับการวิเคราะห์สัญญาณ
    valid_idx = ~np.isnan(v_values)
    v_clean = v_values[valid_idx]
    t_clean = time_values[valid_idx]
    f_clean = frames[valid_idx]

    # ค้นหาจุดกลับตัว: Peaks (จุดสูงสุด) และ Troughs (จุดต่ำสุด)
    # ใช้ find_peaks โดยคูณสัญญาณด้วย -1 เพื่อหาจุดต่ำสุด
    peaks, _ = find_peaks(v_clean, distance=15, prominence=10)
    troughs, _ = find_peaks(-v_clean, distance=15, prominence=10)

    lift_count = 0
    lift_log = []

    # จับคู่คู่จุดต่ำสุด (ก้ม) และจุดสูงสุด (ยก) เพื่อพิจารณาการยก 1 รอบ
    for tr in troughs:
        # หาจุดยอดที่เกิดขึ้นหลังจากจุดต่ำสุดนี้
        subsequent_peaks = peaks[peaks > tr]
        if len(subsequent_peaks) > 0:
            pk = subsequent_peaks[0]
            diff = v_clean[pk] - v_clean[tr]

            # ถ้าระยะยกห่างกันเกินเกณฑ์ที่กำหนด
            if diff >= v_threshold:
                lift_count += 1
                trigger_frame = f_clean[pk]
                trigger_time = t_clean[pk]
                lift_log.append({"frame": trigger_frame, "time": trigger_time, "V": v_clean[pk]})
                print(f"[LIFT {lift_count}] ตรวจพบรอบการยกสมบูรณ์ที่เวลา {trigger_time:.2f} วินาที (เฟรม {trigger_frame:.0f}) | ระยะยกสตรก: {diff:.1f} ซม.")

    # แสดงกราฟแสดงผล
    plt.figure(figsize=(14, 5))
    plt.plot(results["time_s"], results["V_cm"], label="Hand Height (V_cm)", color="#1a73e8", lw=2.5)

    # แสดงจุดต่ำสุด-สูงสุดที่ตรวจพบ
    plt.plot(t_clean[troughs], v_clean[troughs], "v", color="#ea4335", markersize=10, label="Troughs (จุดต่ำสุดตอนก้ม)")
    plt.plot(t_clean[peaks], v_clean[peaks], "^", color="#34a853", markersize=10, label="Peaks (จุดสูงสุดตอนยก)")

    # ไฮไลต์จุดนับรอบสำเร็จ
    for log in lift_log:
        plt.plot(log["time"], log["V"], "ro", markersize=12, fillstyle='none', mew=2.5, label="Counted Lift")

    plt.title(f"ระบบนับจำนวนครั้งการยกแบบ Robust (ตรวจพบทั้งหมด {lift_count} ครั้ง)")
    plt.xlabel("Time (s)")
    plt.ylabel("V (cm)")
    plt.grid(True, alpha=0.25)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

    return lift_count, lift_log