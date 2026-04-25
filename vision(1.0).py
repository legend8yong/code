from maix import image, camera, display, app, uart, pinmap
import cv2
import math
import numpy as np
import time


# UART1: A19(TX), A18(RX), device /dev/ttyS1.
# If your board exposes UART1 on different pins, only change these mappings.
pinmap.set_pin_function("A18", "UART1_RX")
pinmap.set_pin_function("A19", "UART1_TX")
UART_DEVICE = "/dev/ttyS1"
UART_BAUD = 115200

FRAME_W = 320
FRAME_H = 240
DETECT_X1 = FRAME_W // 4
DETECT_X2 = FRAME_W * 3 // 4
DETECT_W = DETECT_X2 - DETECT_X1
DETECT_Y1 = 0
DETECT_Y2 = FRAME_H * 2 // 3
DETECT_H = DETECT_Y2 - DETECT_Y1

MIN_AREA = 350
MAX_AREA_RATIO = 0.65
MAX_AREA = DETECT_W * DETECT_H * MAX_AREA_RATIO
UART_BURST_MS = 1800
UART_REPEAT_INTERVAL_MS = 120

# HSV thresholds for RGB image converted by cv2.COLOR_RGB2HSV.
# Adjust S/V lower bounds if the light is weak or the object color is pale.
RED_RANGES = (
    ((0, 70, 70), (8, 255, 255)),
    ((172, 70, 70), (179, 255, 255)),
)
GREEN_RANGES = (
    ((32, 40, 40), (95, 255, 255)),
)
REFLECTION_LOW = (0, 0, 180)
REFLECTION_HIGH = (179, 85, 255)

# Looser thresholds tolerate perspective skew: a square may look like a
# trapezoid/rhombus when the camera is tilted relative to the target plane.
SQUARE_SIDE_RATIO_LIMIT = 1.80
RIGHT_ANGLE_COS_LIMIT = 0.65
CIRCLE_CIRCULARITY_MIN = 0.65
CIRCLE_VERTEX_MIN = 6
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
REFLECTION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

serial1 = uart.UART(UART_DEVICE, UART_BAUD)
cam = camera.Camera(FRAME_W, FRAME_H)
disp = display.Display()

uart_msg = None
uart_burst_end_ms = 0
last_send_ms = 0
last_frame_ts = time.time()
fps_smooth = 0.0


def ticks_ms():
    return int(time.time() * 1000)


def point_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


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
    shape_contour = cv2.convexHull(contour)
    area = cv2.contourArea(shape_contour)
    if area < MIN_AREA:
        return None, None

    perimeter = cv2.arcLength(shape_contour, True)
    if perimeter <= 0:
        return None, None

    approx = cv2.approxPolyDP(shape_contour, 0.035 * perimeter, True)
    points = [tuple(p[0]) for p in approx]

    if cv2.isContourConvex(approx) and is_square(points):
        return "Square", approx

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    if len(points) >= CIRCLE_VERTEX_MIN and circularity >= CIRCLE_CIRCULARITY_MIN:
        return "Circle", approx

    return None, None


def build_range_mask(hsv, ranges):
    mask = None
    for low, high in ranges:
        part = cv2.inRange(hsv, low, high)
        if mask is None:
            mask = part
        else:
            mask = cv2.bitwise_or(mask, part)

    return mask


def build_reflection_mask(hsv, color_mask):
    reflection_mask = cv2.inRange(hsv, REFLECTION_LOW, REFLECTION_HIGH)
    near_color_mask = cv2.dilate(color_mask, REFLECTION_KERNEL, iterations=1)
    return cv2.bitwise_and(reflection_mask, near_color_mask)


def build_target_mask(hsv, red_mask, green_mask):
    mask = cv2.bitwise_or(red_mask, green_mask)
    reflection_mask = build_reflection_mask(hsv, mask)
    mask = cv2.bitwise_or(mask, reflection_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, REFLECTION_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    return mask


def classify_color(contour, red_mask, green_mask):
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None

    contour_roi = contour - (x, y)
    fill_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(fill_mask, [contour_roi], -1, 255, -1)

    red_roi = red_mask[y : y + h, x : x + w]
    green_roi = green_mask[y : y + h, x : x + w]
    red_count = cv2.countNonZero(cv2.bitwise_and(red_roi, fill_mask))
    green_count = cv2.countNonZero(cv2.bitwise_and(green_roi, fill_mask))

    if red_count <= 0 and green_count <= 0:
        return None
    return "Red" if red_count >= green_count else "Green"


def find_targets(mask, red_mask, green_mask, x_offset=0, y_offset=0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    targets = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        shape_name, approx = classify_shape(contour)
        if shape_name is None:
            continue

        color_name = classify_color(contour, red_mask, green_mask)
        if color_name is None:
            continue

        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue

        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        if x_offset or y_offset:
            approx = approx + (x_offset, y_offset)
            cx += x_offset
            cy += y_offset

        targets.append(
            {
                "shape": shape_name,
                "color": color_name,
                "area": area,
                "center": (cx, cy),
                "approx": approx,
            }
        )

    return targets


def update_uart(shape_name=None, color_name=None):
    global uart_msg, uart_burst_end_ms, last_send_ms

    now_ms = ticks_ms()
    if uart_msg is not None and now_ms >= uart_burst_end_ms:
        uart_msg = None

    if uart_msg is None:
        if shape_name is None or color_name is None:
            return
        uart_msg = "S:{},C:{}".format(shape_name, color_name)
        uart_burst_end_ms = now_ms + UART_BURST_MS
        last_send_ms = now_ms - UART_REPEAT_INTERVAL_MS

    if now_ms - last_send_ms < UART_REPEAT_INTERVAL_MS:
        return

    serial1.write_str(uart_msg + "\n")
    last_send_ms = now_ms


while not app.need_exit():
    now = time.time()
    dt = now - last_frame_ts
    last_frame_ts = now
    if dt > 0:
        fps_now = 1.0 / dt
        fps_smooth = fps_now if fps_smooth == 0.0 else fps_smooth * 0.9 + fps_now * 0.1

    img = cam.read()
    img_cv = image.image2cv(img, ensure_bgr=False, copy=False)
    detect_roi = img_cv[DETECT_Y1:DETECT_Y2, DETECT_X1:DETECT_X2]
    hsv = cv2.cvtColor(detect_roi, cv2.COLOR_RGB2HSV)

    red_mask = build_range_mask(hsv, RED_RANGES)
    green_mask = build_range_mask(hsv, GREEN_RANGES)
    target_mask = build_target_mask(hsv, red_mask, green_mask)

    targets = find_targets(target_mask, red_mask, green_mask, DETECT_X1, DETECT_Y1)

    cv2.rectangle(
        img_cv,
        (DETECT_X1, DETECT_Y1),
        (DETECT_X2 - 1, DETECT_Y2 - 1),
        (90, 90, 90),
        1,
    )

    detected_shape_name = None
    detected_color_name = None

    if targets:
        best = max(targets, key=lambda item: item["area"])
        shape_name = best["shape"]
        color_name = best["color"]
        detected_shape_name = shape_name
        detected_color_name = color_name
        cx, cy = best["center"]
        draw_color = (255, 0, 0) if color_name == "Red" else (0, 255, 0)

        cv2.drawContours(img_cv, [best["approx"]], -1, draw_color, 2)
        cv2.drawMarker(img_cv, (cx, cy), draw_color, cv2.MARKER_CROSS, 18, 2)
        cv2.putText(
            img_cv,
            "{} {}".format(color_name, shape_name),
            (max(0, cx - 55), max(20, cy - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            draw_color,
            2,
        )

    update_uart(detected_shape_name, detected_color_name)

    cv2.putText(
        img_cv,
        "FPS:{:.1f}".format(fps_smooth),
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    disp.show(img)
