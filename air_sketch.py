import cv2
import numpy as np
import mediapipe as mp

# ===============================
# CONFIG
# ===============================
PINCH_THRESHOLD = 4
SMOOTHING = 0.7
BRUSH_COLOR = (0, 0, 255)
BRUSH_SIZE = 8

BTN_X1, BTN_Y1 = 20, 20
BTN_X2, BTN_Y2 = 200, 80   # slightly bigger button (easier click)


# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ===============================
# UTIL FUNCTIONS
# ===============================
def create_canvas(frame):
    return np.zeros_like(frame)


def draw_ui(frame):
    cv2.rectangle(frame, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), (0, 255, 0), 2)
    cv2.putText(frame, "CLEAR", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def get_finger_positions(hand_landmarks, shape):
    h, w, _ = shape

    ix = int(hand_landmarks.landmark[8].x * w)   # index tip
    iy = int(hand_landmarks.landmark[8].y * h)

    tx = int(hand_landmarks.landmark[4].x * w)   # thumb tip
    ty = int(hand_landmarks.landmark[4].y * h)

    return ix, iy, tx, ty


def detect_pinch(ix, iy, tx, ty, width):
    dist = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5
    return dist < (0.045 * width)


def smooth(prev, curr):
    if prev is None:
        return curr
    px, py = prev
    cx, cy = curr
    x = int(SMOOTHING * px + (1 - SMOOTHING) * cx)
    y = int(SMOOTHING * py + (1 - SMOOTHING) * cy)
    return x, y


def on_clear_button(x, y):
    return BTN_X1 < x < BTN_X2 and BTN_Y1 < y < BTN_Y2


# ===============================
# MAIN APP
# ===============================
def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not opening. Try index 1 or close other apps using camera.")
        return

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    canvas = None
    prev_point = None

    pinch_frames = 0
    was_pinching = False   # debounce for clear button

    print("✅ Air Sketch Started")
    print("Press ESC or Q to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = create_canvas(frame)

        draw_ui(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pinch = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                ix, iy, tx, ty = get_finger_positions(hand_landmarks, frame.shape)

                # ---------- pinch detection ----------
                if detect_pinch(ix, iy, tx, ty, frame.shape[1]):
                    pinch_frames += 1
                else:
                    pinch_frames = 0

                pinch = pinch_frames >= PINCH_THRESHOLD

                # ---------- CLEAR BUTTON (fixed + reliable) ----------
                if pinch and not was_pinching:
                    if on_clear_button(ix, iy):
                        canvas = create_canvas(frame)
                        prev_point = None

                # ---------- DRAW ----------
                if pinch and not on_clear_button(ix, iy):
                    curr = smooth(prev_point, (ix, iy))

                    if prev_point is not None:
                        cv2.line(canvas, prev_point, curr, BRUSH_COLOR, BRUSH_SIZE)

                    prev_point = curr
                else:
                    prev_point = None

                was_pinching = pinch

        # overlay canvas
        output = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        cv2.imshow("Air Sketch", output)

        key = cv2.waitKey(1) & 0xFF

        # ESC or Q to exit
        if key == 27 or key == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    main()
