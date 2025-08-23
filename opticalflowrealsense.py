#oklu
import cv2 as cv
import numpy as np
import time, math
import pyrealsense2 as rs    # <-- RealSense

# ================= AYARLAR =================
TARGET_FPS = 15.0                              # hedef FPS (işleme hızı)
FRAME_W, FRAME_H = 640, 480                    # renk akış çözünürlüğü
MAX_CORNERS = 200                              # takip edilecek max köşe sayısı
MIN_TRACKED = 100                              # yeniden algılama için minimum nokta
REDETECT_EVERY = 25                            # kaç karede bir yeniden tespit
SCALE = 1.0                                    # monokülerde ölçek bilinmez (depth yoksa 1.0)

# Çizim ayarları
DRAW_FLOW = True                               # okla gösterim
DRAW_EVERY_N = 2                               # kaç noktada bir ok çizilsin
ARROW_TIP = 0.3                                # ok ucu oranı

# Kamera matrisi (RealSense'ten güncellenecek)
K = np.eye(3, dtype=np.float64)

# ================ MODÜLLER ================
def detect_features(gray):       # başlangıç noktalarını bulma
    return cv.goodFeaturesToTrack(
        gray, maxCorners=MAX_CORNERS,
        qualityLevel=0.01, minDistance=7, blockSize=7
    )

def to_euler_zyx(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:     # gimbal lock yoksa normal durum:
        roll  = math.degrees(math.atan2(R[2,1], R[2,2]))   # x-ekseni dönüş
        pitch = math.degrees(math.atan2(-R[2,0], sy))      # y-ekseni dönüş
        yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))   # z-ekseni dönüş
    else:
        roll  = math.degrees(math.atan2(-R[1,2], R[1,1]))
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        yaw   = 0.0
    return yaw, pitch, roll

def to_quaternion(R):
    q = np.empty((4,), dtype=np.float64)       # (x,y,z,w)
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2,1] - R[1,2]) * s
        q[1] = (R[0,2] - R[2,0]) * s
        q[2] = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            q[3] = (R[2,1] - R[1,2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            q[3] = (R[0,2] - R[2,0]) / s
            q[0] = (R[0,1] + R[1,0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            q[3] = (R[1,0] - R[0,1]) / s
            q[0] = (R[0,2] + R[2,0]) / s
            q[1] = (R[1,2] + R[2,1]) / s
            q[2] = 0.25 * s
    return tuple(q)  # (x,y,z,w)

def draw_overlay(img, fps, npts, pos, quat, ypr):
    x,y,z = pos
    yaw,pitch,roll = ypr
    qx,qy,qz,qw = quat
    cv.putText(img, f"FPS: {fps:.2f}", (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv.putText(img, f"Tracked points: {npts}", (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv.putText(img, f"Pos: {x:.2f}, {y:.2f}, {z:.2f}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv.putText(img, f"Yaw/Pitch/Roll: {yaw:.1f}/{pitch:.1f}/{roll:.1f}", (10,80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv.putText(img, f"Quat: {qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}", (10,100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

def process_frame(prev_gray, gray, p0, R_wc, t_wc):
    """Bir karelik optical flow ve pose + çizim için inlier çiftleri."""
    p1, st, _ = cv.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None,
        winSize=(15,15),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=1e-3
    )
    if p1 is None:
        return p0, R_wc, t_wc, [], []

    st = st.reshape(-1)
    good_old = p0[st==1].reshape(-1,2)
    good_new = p1[st==1].reshape(-1,2)

    pairs = []
    if len(good_old) >= 8:
        E, mask = cv.findEssentialMat(
            good_old, good_new, K,
            method=cv.RANSAC, prob=0.999, threshold=1.5
        )
        if E is not None:
            inl = mask.ravel().astype(bool)
            if inl.sum() >= 5:
                _, R_est, t_est, _ = cv.recoverPose(E, good_old[inl], good_new[inl], K)
                R_wc = R_est @ R_wc
                t_wc = R_est @ t_wc + t_est * SCALE
                pairs = list(zip(good_old[inl], good_new[inl]))
            else:
                pairs = list(zip(good_old, good_new))
        else:
            pairs = list(zip(good_old, good_new))

    vis_points = [(int(x2), int(y2)) for (x2,y2) in good_new]
    return good_new.reshape(-1,1,2), R_wc, t_wc, vis_points, pairs

# ================ ANA PROGRAM ================
def main():
    global K

    # ---- RealSense pipeline ----
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)  # renk
    # (İstersen derinliği de açabilirsin: cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30))
    profile = pipeline.start(cfg)

    # Color intrinsics (fx, fy, cx, cy) + distorsiyon (k1..k5)
    s = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = s.get_intrinsics()
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,   1]], dtype=np.float64)
    D = np.array(intr.coeffs[:5], dtype=np.float64)  # [k1, k2, p1, p2, k3] (Brown-Conrady)

    # Undistort/rectify haritaları (bir kez)
    map1, map2 = cv.initUndistortRectifyMap(K, D, None, K, (FRAME_W, FRAME_H), cv.CV_16SC2)

    # İlk kare
    frames = pipeline.wait_for_frames()
    color  = frames.get_color_frame()
    if not color:
        print("İlk kare alınamadı."); pipeline.stop(); return
    frame0 = np.asanyarray(color.get_data())
    frame0 = cv.remap(frame0, map1, map2, cv.INTER_LINEAR)   # distorsiyonsuz renk
    prev_gray = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
    p0 = detect_features(prev_gray)

    # başlangıç poz
    R_wc = np.eye(3, dtype=np.float64)          # Dünya->Kamera dönüş
    t_wc = np.zeros((3,1), dtype=np.float64)    # Dünya->Kamera öteleme

    # FPS kontrol
    period = 1.0 / TARGET_FPS
    last_proc = time.perf_counter()
    last_fps_t = last_proc
    proc_counter = 0
    frames_since_redetect = 0

    while True:
        # Kare al
        frames = pipeline.wait_for_frames()
        color  = frames.get_color_frame()
        if not color:
            if cv.waitKey(1) & 0xFF == ord('q'): break
            continue
        frame_raw = np.asanyarray(color.get_data())
        frame = cv.remap(frame_raw, map1, map2, cv.INTER_LINEAR)  # undistort edilmiş renk

        now = time.perf_counter()
        if (now - last_proc) < period:      # hedef FPS düzenlemesi
            if cv.waitKey(1) & 0xFF == ord('q'): break
            continue
        last_proc = now; proc_counter += 1; frames_since_redetect += 1

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Özellikleri gerektiğinde yenile
        if p0 is None or len(p0) < MIN_TRACKED or (frames_since_redetect % REDETECT_EVERY == 0):
            p0 = detect_features(prev_gray); frames_since_redetect = 0

        # Poz güncelle
        if p0 is not None and len(p0) >= 8:
            p0, R_wc, t_wc, vis_points, flow_pairs = process_frame(prev_gray, gray, p0, R_wc, t_wc)
        else:
            vis_points, flow_pairs = [], []

        # Camera->World
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        pos = (float(t_cw[0]), float(t_cw[1]), float(t_cw[2]))
        quat = to_quaternion(R_cw)
        ypr  = to_euler_zyx(R_cw)

        # Çizim
        vis = frame.copy()
        for pt in vis_points:
            cv.circle(vis, pt, 2, (0,0,255), -1)
        if DRAW_FLOW and flow_pairs:
            for ((x1,y1),(x2,y2)) in flow_pairs[::DRAW_EVERY_N]:
                cv.arrowedLine(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 1, tipLength=ARROW_TIP)

        fps_now = proc_counter / max(now - last_fps_t, 1e-6)
        draw_overlay(vis, fps_now, len(vis_points), pos, quat, ypr)
        cv.imshow("OpticalFlow Pose (Optimized)", vis)

        prev_gray = gray
        if p0 is not None: p0 = p0.reshape(-1,1,2)

        if proc_counter % 30 == 0:
            last_fps_t = now; proc_counter = 0

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): R_wc[:] = np.eye(3); t_wc[:] = 0

    # kaynakları kapat
    pipeline.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
