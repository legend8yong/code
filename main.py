from maix import image, camera, display, app, uart, pinmap
import cv2
import math
import time

pinmap.set_pin_function("A18", "UART1_RX")
pinmap.set_pin_function("A19", "UART1_TX")
UART_DEVICE = "/dev/ttyS1"  # UART1 -> A19(TX), A18(RX)
UART_BAUD = 115200
FRAME_W = 320
FRAME_H = 240

MIN_AREA = 120
MAX_AREA_RATIO = 0.5
BLACK_MAX_GRAY = 70

TRI_SIDE_RATIO_LIMIT = 1.18
SQUARE_SIDE_RATIO_LIMIT = 1.15
RIGHT_ANGLE_COS_LIMIT = 0.25
CIRCLE_CIRCULARITY_MIN = 0.78

ser = uart.UART(UART_DEVICE, UART_BAUD)
cam = camera.Camera(FRAME_W, FRAME_H)
disp = display.Display()
last_frame_ts = time.time()
fps_smooth = 0.0


def point_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_equilateral_triangle(points):
    if len(points) != 3:
        return False
    sides = [
        point_dist(points[0], points[1]),
        point_dist(points[1], points[2]),
        point_dist(points[2], points[0]),
    ]
    min_side = min(sides)
    max_side = max(sides)
    return min_side > 0 and (max_side / min_side) <= TRI_SIDE_RATIO_LIMIT


def is_square(points):
    if len(points) != 4:
        return False

    sides = [point_dist(points[i], points[(i + 1) % 4]) for i in range(4)]
    min_side = min(sides)
    max_side = max(sides)
    if min_side <= 0 or (max_side / min_side) > SQUARE_SIDE_RATIO_LIMIT:
        return False

    for i in range(4):
        p_prev = points[(i - 1) % 4]
        p_curr = points[i]
        p_next = points[(i + 1) % 4]

        v1x, v1y = p_prev[0] - p_curr[0], p_prev[1] - p_curr[1]
        v2x, v2y = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 == 0 or n2 == 0:
            return False

        cos_theta = abs((v1x * v2x + v1y * v2y) / (n1 * n2))
        if cos_theta > RIGHT_ANGLE_COS_LIMIT:
            return False

    return True


def classify_shape(contour):
    area = cv2.contourArea(contour)
    if area < MIN_AREA:
        return None, None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None, None

    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    points = [tuple(p[0]) for p in approx]
    vertex_count = len(points)

    if vertex_count == 3 and is_equilateral_triangle(points):
        return "Triangle", approx
    if vertex_count == 4 and is_square(points):
        return "Square", approx

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    if vertex_count >= 5 and circularity >= CIRCLE_CIRCULARITY_MIN:
        return "Circle", approx

    return None, None


while not app.need_exit():
    now = time.time()
    dt = now - last_frame_ts
    last_frame_ts = now
    if dt > 0:
        fps_now = 1.0 / dt
        if fps_smooth == 0.0:
            fps_smooth = fps_now
        else:
            fps_smooth = fps_smooth * 0.9 + fps_now * 0.1

    img = cam.read()
    img_cv = image.image2cv(img, ensure_bgr=False, copy=False)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_cnt)
        max_area = FRAME_W * FRAME_H * MAX_AREA_RATIO

        if MIN_AREA < area < max_area:
            shape_name, approx = classify_shape(max_cnt)
            if shape_name is not None:
                m = cv2.moments(max_cnt)
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])

                    h, w = gray.shape
                    x0 = max(0, cx - 2)
                    x1 = min(w, cx + 3)
                    y0 = max(0, cy - 2)
                    y1 = min(h, cy + 3)
                    roi = gray[y0:y1, x0:x1]

                    if roi.size > 0:
                        avg_gray = float(cv2.mean(roi)[0])
                        if avg_gray <= BLACK_MAX_GRAY:
                            msg = f"S:{shape_name},C:Black\n"
                            ser.write_str(msg)

                            cv2.drawContours(img_cv, [approx], -1, (0, 255, 0), 2)
                            cv2.drawMarker(
                                img_cv,
                                (cx, cy),
                                (0, 255, 0),
                                cv2.MARKER_CROSS,
                                20,
                                2,
                            )
                            cv2.putText(
                                img_cv,
                                f"{shape_name} (Black)",
                                (cx + 12, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

    cv2.putText(
        img_cv,
        "FPS:{:.1f}".format(fps_smooth),
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    disp.show(img)
