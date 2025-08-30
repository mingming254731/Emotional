import cv2
from deepface import DeepFace

# -------------------- ตั้งค่าทั่วไป --------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

# Smoothing params
ALPHA = 0.4          # 0.2–0.5: ยิ่งต่ำยิ่งนิ่ง
MARGIN = 8.0         # ชนะอันดับสองเกิน MARGIN% ให้สลับทันที
ABS_THRESHOLD = 55.0 # ถ้าคะแนนสูงกว่าเท่านี้ ให้สลับทันที
SWITCH_N = 3         # ถ้าชนะไม่ขาด ต้องชนะติดกันกี่เฟรมถึงจะสลับ

WANTED = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"]

# -------------------- ตัวช่วย --------------------
def analyze_safe(frame):
    """
    ลองใช้ retinaface ก่อน (แม่นกว่า) ถ้าไม่ได้ค่อยตกไป opencv
    รองรับ deepface เวอร์ชันที่ signature ต่างกัน
    """
    for backend in ("retinaface", "opencv"):
        try:
            return DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend=backend,
                enforce_detection=False,
                prog_bar=False
            )
        except TypeError:
            # deepface รุ่นเก่าไม่รับบาง keyword -> เรียกแบบมินิมอล
            return DeepFace.analyze(frame, actions=['emotion'])
        except Exception:
            continue
    # สุดทางแล้วค่อยโยนด้วยแบบมินิมอล
    return DeepFace.analyze(frame, actions=['emotion'])

# -------------------- main --------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    ema_scores = None
    last_dominant = None
    stable_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ไม่พบสัญญาณกล้อง")
            break

        try:
            result = analyze_safe(frame)
            first = result[0] if isinstance(result, list) else result

            # รองรับคีย์ต่างชื่อระหว่างรุ่น
            raw_scores = first.get('emotion') or first.get('emotions') or {}
            cur = {k: float(raw_scores.get(k, 0.0)) for k in WANTED}

            # ---------------- EMA smoothing ----------------
            if ema_scores is None:
                ema_scores = cur.copy()
            else:
                for k in WANTED:
                    ema_scores[k] = ALPHA * cur[k] + (1 - ALPHA) * ema_scores[k]

            # ---------------- ตัดสินใจ dominant ----------------
            # ใช้ค่าจาก EMA เพื่อความนิ่ง
            sorted_emos = sorted(ema_scores.items(), key=lambda x: -x[1])
            cur_dominant, cur_top = sorted_emos[0]
            second_best = sorted_emos[1][1] if len(sorted_emos) > 1 else 0.0

            # เงื่อนไขสลับทันที
            force_switch = (cur_top - second_best) >= MARGIN or cur_top >= ABS_THRESHOLD

            if last_dominant is None:
                last_dominant = cur_dominant
                stable_count = 1
            else:
                if cur_dominant == last_dominant:
                    stable_count += 1
                elif force_switch:
                    last_dominant = cur_dominant
                    stable_count = 1
                else:
                    stable_count += 1
                    if stable_count >= SWITCH_N:
                        last_dominant = cur_dominant
                        stable_count = 1

            show_dom = last_dominant

            # ---------------- วาดผลลัพธ์ ----------------
            # 1) ข้อความ Emotion หลัก
            cv2.putText(frame, f"Emotion: {show_dom}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

            # 2) แสดงคะแนนอารมณ์ (เรียงจากมากไปน้อย)
            y = 80
            for emo, val in sorted(ema_scores.items(), key=lambda x: -x[1]):
                cv2.putText(frame, f"{emo}: {val:.2f}%", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
                y += 26

            # 3) วาดกรอบหน้า (ถ้ามี region)
            region = first.get('region') or {}
            x, y0, w, h = (region.get('x'), region.get('y'), region.get('w'), region.get('h'))
            if all(isinstance(v, (int, float)) for v in [x, y0, w, h]):
                x, y0, w, h = int(x), int(y0), int(w), int(h)
                cv2.rectangle(frame, (x, y0), (x + w, y0 + h), (0, 200, 255), 2)

        except Exception as e:
            cv2.putText(frame, f"DeepFace error: {e}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow("Emotion Detector (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
