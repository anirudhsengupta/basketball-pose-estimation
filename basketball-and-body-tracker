import cv2
import mediapipe as mp
import numpy as np

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video capture
cap = cv2.VideoCapture(0)

# Helper functions
def clamp_int(x, min_val=1):
    return max(min_val, int(x))

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# Main loop
with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # -------- Ball Detection --------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([4, 100, 80])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_pos = None
        ball_radius = 0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area > 800:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True)**2)
                if 0.4 < circularity < 1.4:
                    ball_pos = (int(x), int(y))
                    ball_radius = int(radius * 1.0)
                    cv2.circle(frame, ball_pos, ball_radius, (0, 140, 255), 3)

        # -------- Pose Detection --------
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # -------- 3D Mannequin Simulation --------
        sim = np.zeros_like(frame)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            points = {}
            for i, landmark in enumerate(lm):
                if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                    x, y, z = int(landmark.x * w), int(landmark.y * h), landmark.z * w
                    points[i] = (x, y, z)

            # Draw torso polygon
            torso_idxs = [
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_HIP.value,
                mp_pose.PoseLandmark.LEFT_HIP.value
            ]
            if all(idx in points for idx in torso_idxs):
                pts = np.array([[points[idx][0], points[idx][1]] for idx in torso_idxs], np.int32)
                cv2.fillPoly(sim, [pts], (200, 180, 150))

            # Draw limbs as thick solid lines
            for conn in mp_pose.POSE_CONNECTIONS:
                if conn[0] in points and conn[1] in points:
                    x1, y1, z1 = points[conn[0]]
                    x2, y2, z2 = points[conn[1]]
                    avg_z = (z1 + z2) / 2
                    thickness = clamp_int(8 * (1.2 - avg_z / 1000))
                    color_val = max(50, min(255, int(255 - abs(avg_z) * 50)))
                    color = (color_val, int(color_val * 0.8), 255)
                    cv2.line(sim, (x1, y1), (x2, y2), color, thickness)

            # Draw joints as smaller spheres
            for x, y, z in points.values():
                radius = clamp_int(5 * (1.2 - z / 1000))  # smaller than before
                base_color = np.array([255, 200, 180], dtype=np.float32)
                light_intensity = max(0.3, 1.0 - z / 1000)
                shaded_color = tuple(clamp_int(c * light_intensity) for c in base_color)
                cv2.circle(sim, (x, y), radius, shaded_color, -1)
                cv2.circle(sim, (x - radius//3, y - radius//3), clamp_int(radius//2), (255, 255, 255), -1)

            # -------- Right Elbow Angle --------
            try:
                r_shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value][:2]
                r_elbow = points[mp_pose.PoseLandmark.LEFT_ELBOW.value][:2]
                r_wrist = points[mp_pose.PoseLandmark.LEFT_WRIST.value][:2]
                angle = int(calculate_angle(r_shoulder, r_elbow, r_wrist))
                # Display near right elbow
                text_pos = (r_elbow[0]+10, r_elbow[1]-10)
                cv2.putText(sim, f'{angle} deg', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(sim, f'{angle} deg', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
            except KeyError:
                pass

        # Draw solid orange ball
        if ball_pos:
            bx, by = ball_pos
            cv2.circle(sim, (bx, by), ball_radius, (0, 165, 255), -1)  # single-color orange

        # Combine live + simulation
        combined = np.hstack((frame, sim))
        cv2.imshow("3D Android Mannequin + Ball Simulation", combined)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

cap.release()
cv2.destroyAllWindows()
