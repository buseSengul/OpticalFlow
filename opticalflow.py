import cv2 as cv
import numpy as np
import time, math
from scipy.spatial.transform import Rotation as RScipy


class OpticalFlow:
    def __init__(self,
                 cam_index=0,
                 target_fps=15.0,
                 frame_w=640,
                 frame_h=480,
                 max_corners=200,
                 min_tracked=100,
                 redetect_every=25,
                 scale=1.0):
        
        # ================= AYARLAR =================
        self.CAM_INDEX = cam_index                           # kamera indeksi
        self.TARGET_FPS = target_fps                         # hedef FPS
        self.FRAME_W = frame_w                               # görüntü genişliği
        self.FRAME_H = frame_h                               # görüntü yüksekliği
        self.MAX_CORNERS = max_corners                       # takip edilecek maksimum köşe sayısı
        self.MIN_TRACKED = min_tracked                       # yeniden algılama öncesi minimum nokta sayısı
        self.REDETECT_EVERY = redetect_every                  # kaç karede bir yeniden özellik algılansın
        self.SCALE = scale                                   # poz tahmininde mesafe ölçeği (kalibrasyon gerek)

        # Çizim ayarları
        self.DRAW_FLOW = True                                # optical flow yönünü ok ile göster
        self.DRAW_EVERY_N = 2                                # her kaç noktada bir ok çizilsin
        self.ARROW_TIP = 0.3                                 # ok uzunluğu oranı

        # Kamera matrisi (kalibrasyon sonrası güncelle)
        fx = 350.0
        fy = 350.0
        cx = self.FRAME_W / 2
        cy = self.FRAME_H / 2
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=np.float64)      # kamera iç parametre matrisi (fx, fy, cx, cy)

    # ================= MODÜLLER =================
    def detect_features(self, gray):
        """Başlangıç köşe noktalarını tespit et."""
        return cv.goodFeaturesToTrack(
            gray, maxCorners=self.MAX_CORNERS,
            qualityLevel=0.01, minDistance=7, blockSize=7
        )

    def to_euler_zyx(self, R):
        """Dönüş matrisinden (Z-Y-X) Euler açılarını hesapla (yaw, pitch, roll)."""
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:  # gimbal lock yoksa
            roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))   # x ekseni dönüş
            pitch = math.degrees(math.atan2(-R[2, 0], sy))      # y ekseni dönüş
            yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))    # z ekseni dönüş
        else:  # gimbal lock durumu
            roll = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
            pitch = math.degrees(math.atan2(-R[2, 0], sy))
            yaw = 0.0
        return yaw, pitch, roll

    def draw_overlay(self, img, fps, npts, pos, quat, ypr):
        """Görüntü üzerine FPS, takip noktası sayısı, pozisyon, Euler ve quaternion bilgilerini çiz."""
        x, y, z = pos
        yaw, pitch, roll = ypr
        qx, qy, qz, qw = quat
        cv.putText(img, f"FPS: {fps:.2f}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(img, f"Tracked points: {npts}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(img, f"Pos: {x:.2f}, {y:.2f}, {z:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(img, f"Yaw/Pitch/Roll: {yaw:.1f}/{pitch:.1f}/{roll:.1f}", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(img, f"Quat: {qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def process_frame(self, prev_gray, gray, p0, R_wc, t_wc):
        """Bir karelik optical flow ve pose + çizim için inlier çiftleri."""
        p1, st, _ = cv.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
            minEigThreshold=1e-3
        )
        if p1 is None:
            return p0, R_wc, t_wc, [], []

        st = st.reshape(-1)
        good_old = p0[st == 1].reshape(-1, 2)
        good_new = p1[st == 1].reshape(-1, 2)

        pairs = []
        if len(good_old) >= 8:
            E, mask = cv.findEssentialMat(
                good_old, good_new, self.K,
                method=cv.RANSAC, prob=0.999, threshold=1.5
            )
            if E is not None:
                inl = mask.ravel().astype(bool)
                if inl.sum() >= 5:
                    _, R_est, t_est, _ = cv.recoverPose(E, good_old[inl], good_new[inl], self.K)
                    R_wc = R_est @ R_wc
                    t_wc = R_est @ t_wc + t_est * self.SCALE
                    pairs = list(zip(good_old[inl], good_new[inl]))
                else:
                    pairs = list(zip(good_old, good_new))
            else:
                pairs = list(zip(good_old, good_new))

        vis_points = [(int(x2), int(y2)) for (x2, y2) in good_new]
        return good_new.reshape(-1, 1, 2), R_wc, t_wc, vis_points, pairs

    def run(self):
        cap = cv.VideoCapture(self.CAM_INDEX)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        cap.set(cv.CAP_PROP_FPS, 15)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("Kamera açılamadı.")
            return

        ok, prev = cap.read()
        if not ok:
            print("İlk kare alınamadı.")
            return
        prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        p0 = self.detect_features(prev_gray)

        R_wc = np.eye(3, dtype=np.float64)
        t_wc = np.zeros((3, 1), dtype=np.float64)

        period = 1.0 / self.TARGET_FPS
        last_proc = time.perf_counter()
        last_fps_t = last_proc
        proc_counter = 0
        frames_since_redetect = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.perf_counter()
            if (now - last_proc) < period:
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            last_proc = now
            proc_counter += 1
            frames_since_redetect += 1

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if p0 is None or len(p0) < self.MIN_TRACKED or (frames_since_redetect % self.REDETECT_EVERY == 0):
                p0 = self.detect_features(prev_gray)
                frames_since_redetect = 0

            if p0 is not None and len(p0) >= 8:
                p0, R_wc, t_wc, vis_points, flow_pairs = self.process_frame(prev_gray, gray, p0, R_wc, t_wc)
            else:
                vis_points, flow_pairs = [], []

            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            tx, ty, tz = t_cw.ravel()
            pos = (float(tx), float(ty), float(tz))

            ypr = self.to_euler_zyx(R_cw)
            quat = tuple(RScipy.from_matrix(R_cw).as_quat())

            vis = frame.copy()

            # Nokta çizimi pasif
            # for pt in vis_points:
            #     cv.circle(vis, pt, 2, (0, 0, 255), -1)

            # Ok çizimi pasif
            # if self.DRAW_FLOW and flow_pairs:
            #     for ((x1, y1), (x2, y2)) in flow_pairs[::self.DRAW_EVERY_N]:
            #         cv.arrowedLine(vis, (int(x1), int(y1)), (int(x2), int(y2)),
            #                        (0, 255, 0), 1, tipLength=self.ARROW_TIP)

            fps_now = proc_counter / max(now - last_fps_t, 1e-6)
            self.draw_overlay(vis, fps_now, len(vis_points), pos, quat, ypr)
            cv.imshow("OpticalFlow Pose (Optimized)", vis)

            prev_gray = gray
            if p0 is not None:
                p0 = p0.reshape(-1, 1, 2)

            if proc_counter % 30 == 0:
                last_fps_t = now
                proc_counter = 0

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                R_wc[:] = np.eye(3)
                t_wc[:] = 0

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    of = OpticalFlow()
    of.run()






#  #oklu
# import cv2 as cv
# import numpy as np
# import time, math

# from scipy.spatial.transform import Rotation as RScipy
# # ================= AYARLAR =================
# CAM_INDEX = 0                                  #kamera indeksi
# TARGET_FPS = 15.0                              #hedef fps
# FRAME_W, FRAME_H = 640, 480                    #görüntü boyutu
# MAX_CORNERS = 200                              #takip edilecek max köşe sayısı
# MIN_TRACKED = 100                              #Yeniden algılamadan önce izlenmesi gereken minimum nokta sayısı
# REDETECT_EVERY = 25                            #Kaç karede bir yeniden özellik algılansın
# SCALE = 1.0                                    #Poz tahmininde mesafe ölçeği (kalibrasyon gerek)

# # Çizim ayarları
# DRAW_FLOW = True                               #okla gösterim
# DRAW_EVERY_N = 2                               #kaç noktada bir ok çizilsin
# ARROW_TIP = 0.3                                #ok uzunluğu

# # Kamera matrisi (kalibrasyon sonrası güncelle)**************
# fx = 350.0                               
# fy = 350.0   
# cx = FRAME_W/2
# cy =  FRAME_H/2
# K = np.array([[fx, 0, cx],        #kamera iç parametre matrisi   (fx,fy,cx,cy)
#               [0, fy, cy],
#               [0, 0, 1]], dtype=np.float64)

# # ================ MODÜLLER ================
# def detect_features(gray):       #başlangıç noktalarını bulma
#     return cv.goodFeaturesToTrack(
#         gray, maxCorners=MAX_CORNERS,
#         qualityLevel=0.01, minDistance=7, blockSize=7
#     )

# def to_euler_zyx(R):   
#     sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
#     if sy > 1e-6:     #gimbal lock yoksa normal durum:
#         roll  = math.degrees(math.atan2(R[2,1], R[2,2]))   #x ekseninde dönüş
#         pitch = math.degrees(math.atan2(-R[2,0], sy))      #y ekseninde dönüş
#         yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))   #z ekseninde dönüş
#     else:
#         roll  = math.degrees(math.atan2(-R[1,2], R[1,1]))
#         pitch = math.degrees(math.atan2(-R[2,0], sy))
#         yaw   = 0.0
#     return yaw, pitch, roll
    

# def draw_overlay(img, fps, npts, pos, quat, ypr):
#     x,y,z = pos
#     yaw,pitch,roll = ypr
#     qx,qy,qz,qw = quat
#     cv.putText(img, f"FPS: {fps:.2f}", (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#     cv.putText(img, f"Tracked points: {npts}", (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#     cv.putText(img, f"Pos: {x:.2f}, {y:.2f}, {z:.2f}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#     cv.putText(img, f"Yaw/Pitch/Roll: {yaw:.1f}/{pitch:.1f}/{roll:.1f}", (10,80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#     cv.putText(img, f"Quat: {qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}", (10,100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
   

# def process_frame(prev_gray, gray, p0, R_wc, t_wc):   #önceki kare,şimdiki kare, önceki karedeki features, kamera dönüş matrisi, öteleme matrisi
#     """Bir karelik optical flow ve pose + çizim için inlier çiftleri."""
#     p1, st, _ = cv.calcOpticalFlowPyrLK(      #calcOpticalFlowPyrLK: Lucas–Kanade yöntemiyle önceki karedeki noktaların yeni karedeki yerlerini bulur.    p1 yeni karedeki noktların konumu, st ise bulundu mu maskesi
#         prev_gray, gray, p0, None,
#         winSize=(15,15),                  # Lucas-Kanade pencere boyutu
#         maxLevel=2,                     
#         criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 30, 0.01),     #******************30?
#         flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
#         minEigThreshold=1e-3             # zayıf noktaları ele
#     )
#     if p1 is None:
#         return p0, R_wc, t_wc, [], []

#     st = st.reshape(-1)
#     good_old = p0[st==1].reshape(-1,2)
#     good_new = p1[st==1].reshape(-1,2)

#     pairs = []
#     if len(good_old) >= 8:
#         E, mask = cv.findEssentialMat(
#             good_old, good_new, K,
#             method=cv.RANSAC, prob=0.999, threshold=1.5  # gürültüye tolerans
#         )
#         if E is not None:     #kamera pozu güncelleme
#             inl = mask.ravel().astype(bool)
#             if inl.sum() >= 5:
#                 _, R_est, t_est, _ = cv.recoverPose(E, good_old[inl], good_new[inl], K)
#                 R_wc = R_est @ R_wc
#                 t_wc = R_est @ t_wc + t_est * SCALE
#                 pairs = list(zip(good_old[inl], good_new[inl]))
#             else:
#                 pairs = list(zip(good_old, good_new))
#         else:
#             pairs = list(zip(good_old, good_new))

#     vis_points = [(int(x2), int(y2)) for (x2,y2) in good_new]
#     return good_new.reshape(-1,1,2), R_wc, t_wc, vis_points, pairs


# def main():
#     cap = cv.VideoCapture(CAM_INDEX)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_W)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_H)
#     cap.set(cv.CAP_PROP_FPS, 15)
#     cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  #gecikmeyi azaltmak için buffer küçült

#     if not cap.isOpened():
#         print("Kamera açılamadı."); return

#     ok, prev = cap.read()
#     if not ok:
#         print("İlk kare alınamadı."); return
#     prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
#     p0 = detect_features(prev_gray)     #detect_features opencv goodfeaturestodetect  ile ile takip edilecek köşe noktaları tespit et

# #başlangıç poz değerleri
#     R_wc = np.eye(3, dtype=np.float64)          # Dünya->Kamera dönüş matrisi
#     t_wc = np.zeros((3,1), dtype=np.float64)    # Dünya->Kamera öteleme vektörü

# #fps ayarlama
#     period = 1.0 / TARGET_FPS
#     last_proc = time.perf_counter()
#     last_fps_t = last_proc
#     proc_counter = 0
#     frames_since_redetect = 0

#     while True:
#         ok, frame = cap.read()
#         if not ok: break
#         now = time.perf_counter()
#         if (now - last_proc) < period:      #FPS hedefinin altına düşmemesi için belirlenen period kadar beklenir.
#             if cv.waitKey(1) & 0xFF == ord('q'): break
#             continue
#         last_proc = now; proc_counter += 1; frames_since_redetect += 1

#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#         if p0 is None or len(p0) < MIN_TRACKED or (frames_since_redetect % REDETECT_EVERY == 0):
#             p0 = detect_features(prev_gray); frames_since_redetect = 0      #Takip noktası sayısı çok azaldıysa veya belli periyotta (REDETECT_EVERY) tekrar köşe noktaları aranır.

#         if p0 is not None and len(p0) >= 8:     #Yeterli sayıda nokta varsa process_frame() çalıştırılır
#             p0, R_wc, t_wc, vis_points, flow_pairs = process_frame(prev_gray, gray, p0, R_wc, t_wc)
#         else:
#             vis_points, flow_pairs = [], []

#         # Camera pose->World pose
#         R_cw = R_wc.T
#         t_cw = -R_cw @ t_wc
#         #pos = (float(t_cw[0]), float(t_cw[1]), float(t_cw[2]))
#         tx,ty,tz = t_cw.ravel()                    #boyut hatalarını önlemek için ravel ile düzleştir
#         pos = (float(tx),float(ty),float(tz))

#         #Euler (z,y,x) , quat doğrudan matristen
#         ypr  = to_euler_zyx(R_cw)
#         quat = tuple(RScipy.from_matrix(R_cw).as_quat())

#         # Çizim
#         vis = frame.copy()
#         for pt in vis_points:
#             cv.circle(vis, pt, 2, (0,0,255), -1)    #Takip noktalarını kırmızı daire ile işaretler.

#         if DRAW_FLOW and flow_pairs:
#             for ((x1,y1),(x2,y2)) in flow_pairs[::DRAW_EVERY_N]:
#                 cv.arrowedLine(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 1, tipLength=ARROW_TIP)     #Hareket yönünü yeşil oklarla çizer.

#         fps_now = proc_counter / max(now - last_fps_t, 1e-6)
#         draw_overlay(vis, fps_now, len(vis_points), pos, quat, ypr)
#         cv.imshow("OpticalFlow Pose (Optimized)", vis)

#         prev_gray = gray
#         if p0 is not None: p0 = p0.reshape(-1,1,2)

#         if proc_counter % 30 == 0:
#             last_fps_t = now; proc_counter = 0

#         key = cv.waitKey(1) & 0xFF
#         if key == ord('q'): break
#         if key == ord('r'): R_wc[:] = np.eye(3); t_wc[:] = 0

#     cap.release(); cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()

