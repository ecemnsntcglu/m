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

# Renk aralıkları (BGR)
lower_green = np.array([0, 180, 0])
upper_green = np.array([100, 255, 100])

lower_red = np.array([0, 0, 150])
upper_red = np.array([100, 100, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    green_mask = cv2.inRange(frame, lower_green, upper_green)
    red_mask = cv2.inRange(frame, lower_red, upper_red)

    # YEŞİL TOP
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_green:
        (x, y, w, h) = cv2.boundingRect(c)
        obj_left = x
        obj_right = x + w
        label = "green"

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

    # KIRMIZI TOP
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_red:
        (x, y, w, h) = cv2.boundingRect(c)
        obj_left = x
        obj_right = x + w
        label = "red"

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

    # Çizgileri çiz
    cv2.line(frame, (start_line_x, 0), (start_line_x, frame.shape[0]), (0, 0, 0), 2)
    cv2.line(frame, (end_line_x, 0), (end_line_x, frame.shape[0]), (0, 0, 0), 2)

    cv2.imshow("BGR Renk Takip", frame)

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n🟢 YEŞİL TOP TAM TUR: {int(tur_sayisi['green'])}")
print(f"🔴 KIRMIZI TOP TAM TUR: {int(tur_sayisi['red'])}")

df = pd.DataFrame(crossings, columns=["Top", "Çizgi", "Geçiş Zamanı (saniye)"])
print("\nGEÇİŞ TABLOSU:\n")
print(df)
