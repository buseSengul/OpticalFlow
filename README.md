# Optical Flow Tabanlı Kamera Pozu Tahmini

Bu proje, **OpenCV** kullanarak gerçek zamanlı olarak bir kameradan gelen görüntüler üzerinde **optical flow** ile köşe noktaları takibi yapar ve **kamera pozu (konum + yönelim)** tahmini gerçekleştirir.

## Özellikler
- **Shi–Tomasi corner detection** (`cv.goodFeaturesToTrack`) ile özellik noktası tespiti
- **Lucas–Kanade Optical Flow** (`cv.calcOpticalFlowPyrLK`) ile nokta takibi
- **Epipolar Geometri** (`cv.findEssentialMat`) ile kamera hareketini hesaplama
- **Pose Recovery** (`cv.recoverPose`) ile 3B dönüş ve öteleme vektörü çıkarma
- Pozu hem **Euler açıları** hem de **quaternion** formatında gösterme
- İzlenen noktaları ve hareket yönlerini **oklarla görselleştirme**
- FPS ve poz bilgisini ekranda overlay olarak gösterme
- Belirli periyotlarda veya nokta sayısı düştüğünde otomatik yeniden köşe tespiti

---

## Dosya Açıklaması
- `main.py` → Tüm akışın çalıştığı ana Python dosyası.
- İçerdiği ana fonksiyonlar:
  - **`detect_features(gray)`** → Giriş gri görüntüden köşe noktaları tespit eder.
  - **`process_frame(prev_gray, gray, p0, R_wc, t_wc)`** → Optical flow hesaplar, Essential Matrix bulur, poz günceller.
  - **`to_euler_zyx(R)`** → Dönüş matrisinden Euler açılarını hesaplar.
  - **`to_quaternion(R)`** → Dönüş matrisinden quaternion hesaplar.
  - **`draw_overlay(...)`** → FPS, poz ve quaternion bilgisini ekrana yazar.

---

## Parametreler

| Parametre              | Açıklama |
|------------------------|----------|
| `CAM_INDEX`            | Kullanılacak kamera indeksi (0 = varsayılan kamera) |
| `TARGET_FPS`           | Hedef FPS değeri |
| `FRAME_W`, `FRAME_H`   | Kamera çözünürlüğü |
| `MAX_CORNERS`          | Tespit edilecek maksimum köşe sayısı |
| `MIN_TRACKED`          | Yeniden algılama öncesi minimum takip edilen köşe sayısı |
| `REDETECT_EVERY`       | Kaç karede bir otomatik yeniden tespit yapılacağı |
| `SCALE`                | Öteleme vektörü ölçek faktörü |
| `DRAW_FLOW`            | Hareket yönlerini okla çizme seçeneği |
| `DRAW_EVERY_N`         | Kaç noktada bir ok çizileceği |
| `ARROW_TIP`            | Ok uzunluğu |
| `K`                    | Kamera iç parametre matrisi (kalibrasyon sonrası güncellenmeli) |

---

## Akış
1. Kamera açılır ve ilk kare alınır.
2. Köşe noktaları tespit edilir.
3. Sonsuz döngü başlar:
   - Yeni kare alınır.
   - Lucas–Kanade Optical Flow ile noktalar takip edilir.
   - Essential Matrix hesaplanır, ardından kamera pozu çıkarılır.
   - Dünya koordinat sistemindeki poz (Euler + Quaternion) hesaplanır.
   - Noktalar ve hareket yönleri görselleştirilir.
4. Kullanıcı `q` ile çıkabilir, `r` ile poz sıfırlayabilir.

---

### Gereksinimler
```bash
pip install opencv-python numpy
```

## Notlar

- Daha doğru poz tahmini için kamera kalibrasyonu yapıp K matrisini güncelleyin.  

- SCALE değeri sahne ölçeğine göre ayarlanmalıdır.  

- Karanlık veya düşük kontrastlı sahnelerde köşe tespiti zorlaşır.  

- Çok hızlı kamera hareketlerinde inlier sayısı düşebilir, bu durumda poz kararsızlaşır.  
