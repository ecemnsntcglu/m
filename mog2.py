import cv2
import numpy as np
import time
import pandas as pd

# Video yolu
cap = cv2.VideoCapture('C:/Users/Admin/.anaconda/videos/2top.mp4')

ret, frame = cap.read()
if frame is None:
    print("❌ Kare alınamadı.")
    cap.release()
    exit()

start_time = time.time()
start_line_x = 50
end_line_x = 590
offset = 5

crossings = []
previous_state = {"green": "none", "red": "none"}
tur_sayisi = {"green": 0, "red": 0}

fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mask = fgbg.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 400:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        obj_left = x
        obj_right = x + w
        center_y = y + h // 2

        label = "green" if center_y < frame.shape[0] // 2 else "red"

        if abs(obj_left - start_line_x) < offset and previous_state[label] != "start":
            previous_state[label] = "start"
            elapsed = time.time() - start_time
            tur_sayisi[label] += 0.5
            crossings.append((label.upper(), "START", round(elapsed, 2)))
        elif abs(obj_right - end_line_x) < offset and previous_state[label] != "end":
            previous_state[label] = "end"
            elapsed = time.time() - start_time
            tur_sayisi[label] += 0.5
            crossings.append((label.upper(), "END", round(elapsed, 2)))

    # Görselleştirme
    cv2.line(frame, (start_line_x, 0), (start_line_x, frame.shape[0]), (0, 0, 0), 2)
    cv2.line(frame, (end_line_x, 0), (end_line_x, frame.shape[0]), (0, 0, 0), 2)
    cv2.imshow("MOG2 Takip", frame)

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n🟢 YEŞİL TOP TAM TUR: {int(tur_sayisi['green'])}")
print(f"🔴 KIRMIZI TOP TAM TUR: {int(tur_sayisi['red'])}")

df = pd.DataFrame(crossings, columns=["Top", "Çizgi", "Geçiş Zamanı (saniye)"])
print("\nGEÇİŞ TABLOSU:\n")
print(df)
