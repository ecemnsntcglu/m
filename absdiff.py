import cv2
import numpy as np
import time
import pandas as pd

# Video yolu
cap = cv2.VideoCapture('C:/Users/Admin/.anaconda/videos/2top.mp4')

ret, frame1 = cap.read()
ret2, frame2 = cap.read()

if frame1 is None or frame2 is None:
    print("❌ Kare alınamadı. Video bozuk veya yol hatalı.")
    cap.release()
    exit()


start_time = time.time()

start_line_x = 50
end_line_x = 590
offset = 5

# Geçiş verileri (her top için ayrı)
crossings = []

# Nesneleri ayırmak için pozisyonlarına göre etiketleyeceğiz
previous_state = {"green": "none", "red": "none"}

# Tur sayıları (yarım yarım artar)
tur_sayisi = {"green": 0, "red": 0}

while cap.isOpened() and ret and ret2:
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        obj_left = x
        obj_right = x + w
        center_y = y + h // 2

        # Yeşil top üstte, kırmızı altta (y'ye göre ayırıyoruz)
        if center_y < frame1.shape[0] // 2:
            label = "green"
            color = (0, 255, 0)
        else:
            label = "red"
            color = (0, 0, 255)

        # Geçiş kontrolü
        if abs(obj_left - start_line_x) < offset:
            if previous_state[label] != "start":
                previous_state[label] = "start"
                elapsed = time.time() - start_time
                tur_sayisi[label] += 0.5
                crossings.append((label.upper(), "START", round(elapsed, 2)))
                print(f"{label.upper()} TOP - START çizgisi - ZAMAN: {round(elapsed, 2)} sn")

        elif abs(obj_right - end_line_x) < offset:
            if previous_state[label] != "end":
                previous_state[label] = "end"
                elapsed = time.time() - start_time
                tur_sayisi[label] += 0.5
                crossings.append((label.upper(), "END", round(elapsed, 2)))
                print(f"{label.upper()} TOP - END çizgisi - ZAMAN: {round(elapsed, 2)} sn")

        # Takip kutusu çiz
        cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)

    # Çizgileri çiz
    cv2.line(frame1, (start_line_x, 0), (start_line_x, frame1.shape[0]), (0, 0, 0), 2)
    cv2.line(frame1, (end_line_x, 0), (end_line_x, frame1.shape[0]), (0, 0, 0), 2)

    # Göster
    cv2.imshow("Takip", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# TUR SONUÇLARI
print(f"\n🟢 YEŞİL TOP TAM TUR: {int(tur_sayisi['green'])}")
print(f"🔴 KIRMIZI TOP TAM TUR: {int(tur_sayisi['red'])}")

# Pandas tablosu
df = pd.DataFrame(crossings, columns=["Top", "Çizgi", "Geçiş Zamanı (saniye)"])
print("\nGEÇİŞ TABLOSU:\n")
print(df)
