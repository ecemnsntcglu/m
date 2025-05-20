import cv2
import numpy as np
import pandas as pd

# Video yolu
cap = cv2.VideoCapture('C:/Users/Admin/.anaconda/videos/2top.mp4')

ret, frame = cap.read()
if frame is None:
    print("❌ Kare alınamadı. Video bozuk veya yol hatalı.")
    cap.release()
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)  # FPS alınır
frame_no = 0

start_line_x = 50
end_line_x = 590
offset = 5

crossings = []
previous_state = {"green": "none", "red": "none"}
tur_sayisi = {"green": 0, "red": 0}

# HSV renk aralıkları
lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 255, 255])

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 70, 50])
upper_red2 = np.array([180, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Maskeleri oluştur
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Geçerli zaman (saniye cinsinden)
    elapsed = frame_no / fps

    # Konturlar: Yeşil Top
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_green:
        (x, y, w, h) = cv2.boundingRect(contour)
        obj_left = x
        obj_right = x + w
        label = "green"
        color = (0, 255, 0)

        if abs(obj_left - start_line_x) < offset and previous_state[label] != "start":
            previous_state[label] = "start"
            tur_sayisi[label] += 0.5
            crossings.append((label.upper(), "START", round(elapsed, 2)))
            print(f"{label.upper()} TOP - START çizgisi - ZAMAN: {round(elapsed, 2)} sn")

        elif abs(obj_right - end_line_x) < offset and previous_state[label] != "end":
            previous_state[label] = "end"
            tur_sayisi[label] += 0.5
            crossings.append((label.upper(), "END", round(elapsed, 2)))
            print(f"{label.upper()} TOP - END çizgisi - ZAMAN: {round(elapsed, 2)} sn")

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Konturlar: Kırmızı Top
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_red:
        (x, y, w, h) = cv2.boundingRect(contour)
        obj_left = x
        obj_right = x + w
        label = "red"
        color = (0, 0, 255)

        if abs(obj_left - start_line_x) < offset and previous_state[label] != "start":
            previous_state[label] = "start"
            tur_sayisi[label] += 0.5
            crossings.append((label.upper(), "START", round(elapsed, 2)))
            print(f"{label.upper()} TOP - START çizgisi - ZAMAN: {round(elapsed, 2)} sn")

        elif abs(obj_right - end_line_x) < offset and previous_state[label] != "end":
            previous_state[label] = "end"
            tur_sayisi[label] += 0.5
            crossings.append((label.upper(), "END", round(elapsed, 2)))
            print(f"{label.upper()} TOP - END çizgisi - ZAMAN: {round(elapsed, 2)} sn")

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Çizgileri çiz
    cv2.line(frame, (start_line_x, 0), (start_line_x, frame.shape[0]), (0, 0, 0), 2)
    cv2.line(frame, (end_line_x, 0), (end_line_x, frame.shape[0]), (0, 0, 0), 2)

    cv2.imshow("HSV Takip", frame)

    frame_no += 1  # Frame sayacını artır

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
