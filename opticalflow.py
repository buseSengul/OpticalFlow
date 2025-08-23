 #oklu
import cv2 as cv
import numpy as np
import time, math

# ================= AYARLAR =================
CAM_INDEX = 0                                  #kamera indeksi
TARGET_FPS = 15.0                              #hedef fps
FRAME_W, FRAME_H = 640, 480                    #görüntü boyutu
MAX_CORNERS = 200                              #takip edilecek max köşe sayısı
MIN_TRACKED = 100                              #Yeniden algılamadan önce izlenmesi gereken minimum nokta sayısı
REDETECT_EVERY = 25                            #Kaç karede bir yeniden özellik algılansın
SCALE = 1.0                                    #Poz tahmininde mesafe ölçeği (kalibrasyon gerek)

# Çizim ayarları
DRAW_FLOW = True                               #okla gösterim
DRAW_EVERY_N = 2                               #kaç noktada bir ok çizilsin
ARROW_TIP = 0.3                                #ok uzunluğu

# Kamera matrisi (kalibrasyon sonrası güncelle)**************
fx = 350.0                               
fy = 350.0   
cx = FRAME_W/2
cy =  FRAME_H/2
K = np.array([[fx, 0, cx],        #kamera iç parametre matrisi   (fx,fy,cx,cy)
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float64)

# ================ MODÜLLER ================
def detect_features(gray):       #başlangıç noktalarını bulma
    return cv.goodFeaturesToTrack(
        gray, maxCorners=MAX_CORNERS,
        qualityLevel=0.01, minDistance=7, blockSize=7
    )

def to_euler_zyx(R):   
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:     #gimbal lock yoksa normal durum:
        roll  = math.degrees(math.atan2(R[2,1], R[2,2]))   #x ekseninde dönüş
        pitch = math.degrees(math.atan2(-R[2,0], sy))      #y ekseninde dönüş
        yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))   #z ekseninde dönüş
    else:
        roll  = math.degrees(math.atan2(-R[1,2], R[1,1]))
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        yaw   = 0.0
    return yaw, pitch, roll

def to_quaternion(R):
    q = np.empty((4,), dtype=np.float64)       #4 elemanlı x,y,z,w dizi oluştur
    tr = np.trace(R)                           #matris köşegenlerinin toplamı
    if tr > 0:                                 #Trace pozitifse, dönüş matrisindeki açı küçükse veya simetrikse
        s = 0.5 / math.sqrt(tr + 1.0)          #ölçek fakyörü
        q[3] = 0.25 / s
        q[0] = (R[2,1] - R[1,2]) * s
        q[1] = (R[0,2] - R[2,0]) * s
        q[2] = (R[1,0] - R[0,1]) * s
    else:                                      #Trace pozitif değilse, en büyük köşegen elemanı seçilerek hesap yapılır.               
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:    #R[0,0] en büyükse
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            q[3] = (R[2,1] - R[1,2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:                    #R[1,1] en büyükse
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            q[3] = (R[0,2] - R[2,0]) / s
            q[0] = (R[0,1] + R[1,0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1,2] + R[2,1]) / s
        else:                                      #R[2,2] en büyükse
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
   

def process_frame(prev_gray, gray, p0, R_wc, t_wc):   #önceki kare,şimdiki kare, önceki karedeki features, kamera dönüş matrisi, öteleme matrisi
    """Bir karelik optical flow ve pose + çizim için inlier çiftleri."""
    p1, st, _ = cv.calcOpticalFlowPyrLK(      #calcOpticalFlowPyrLK: Lucas–Kanade yöntemiyle önceki karedeki noktaların yeni karedeki yerlerini bulur.    p1 yeni karedeki noktların konumu, st ise bulundu mu maskesi
        prev_gray, gray, p0, None,
        winSize=(15,15),                  # Lucas-Kanade pencere boyutu
        maxLevel=2,                     
        criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 30, 0.01),     #******************30?
        flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=1e-3             # zayıf noktaları ele
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
            method=cv.RANSAC, prob=0.999, threshold=1.5  # gürültüye tolerans
        )
        if E is not None:     #kamera pozu güncelleme
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


def main():
    cap = cv.VideoCapture(CAM_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv.CAP_PROP_FPS, 15)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  #gecikmeyi azaltmak için buffer küçült

    if not cap.isOpened():
        print("Kamera açılamadı."); return

    ok, prev = cap.read()
    if not ok:
        print("İlk kare alınamadı."); return
    prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    p0 = detect_features(prev_gray)     #detect_features opencv goodfeaturestodetect  ile ile takip edilecek köşe noktaları tespit et

#başlangıç poz değerleri
    R_wc = np.eye(3, dtype=np.float64)          # Dünya->Kamera dönüş matrisi
    t_wc = np.zeros((3,1), dtype=np.float64)    # Dünya->Kamera öteleme vektörü

#fps ayarlama
    period = 1.0 / TARGET_FPS
    last_proc = time.perf_counter()
    last_fps_t = last_proc
    proc_counter = 0
    frames_since_redetect = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        now = time.perf_counter()
        if (now - last_proc) < period:      #FPS hedefinin altına düşmemesi için belirlenen period kadar beklenir.
            if cv.waitKey(1) & 0xFF == ord('q'): break
            continue
        last_proc = now; proc_counter += 1; frames_since_redetect += 1

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if p0 is None or len(p0) < MIN_TRACKED or (frames_since_redetect % REDETECT_EVERY == 0):
            p0 = detect_features(prev_gray); frames_since_redetect = 0      #Takip noktası sayısı çok azaldıysa veya belli periyotta (REDETECT_EVERY) tekrar köşe noktaları aranır.

        if p0 is not None and len(p0) >= 8:     #Yeterli sayıda nokta varsa process_frame() çalıştırılır
            p0, R_wc, t_wc, vis_points, flow_pairs = process_frame(prev_gray, gray, p0, R_wc, t_wc)
        else:
            vis_points, flow_pairs = [], []

        # Camera pose->World pose
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        pos = (float(t_cw[0]), float(t_cw[1]), float(t_cw[2]))
        quat = to_quaternion(R_cw)
        ypr  = to_euler_zyx(R_cw)

        # Çizim
        vis = frame.copy()
        for pt in vis_points:
            cv.circle(vis, pt, 2, (0,0,255), -1)    #Takip noktalarını kırmızı daire ile işaretler.

        if DRAW_FLOW and flow_pairs:
            for ((x1,y1),(x2,y2)) in flow_pairs[::DRAW_EVERY_N]:
                cv.arrowedLine(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 1, tipLength=ARROW_TIP)     #Hareket yönünü yeşil oklarla çizer.

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

    cap.release(); cv.destroyAllWindows()

if __name__ == "__main__":
    main()




