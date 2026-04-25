from maix import image, camera, display, app, uart, pinmap
import cv2
import heapq
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
UART_SEND_NAV_INFO = False

MODEL_NODES = ("O", "A", "B", "C", "D", "E", "F", "G", "H")
REGION_NODES = ("A", "B", "C", "D", "E", "F", "G", "H")
TARGET_TOTAL = 4
REGION_BIT = dict((node, 1 << i) for i, node in enumerate(REGION_NODES))
ALL_REGION_MASK = (1 << len(REGION_NODES)) - 1
MODEL_EDGE_CM = 15.0
TURN_EPS = 1e-6
NAV_START_PREVIOUS_NODE = "O"
NAV_START_CURRENT_NODE = "A"
NAV_RETURN_NODE = "O"

# Approximate embedded honeycomb graph from the reference map. Adjust these
# edges/positions if the real track is measured differently.
MAP_POS = {
    "O": (4.0, 0.0),
    "A": (3.0, 0.0),
    "B": (2.5, -0.866),
    "C": (1.5, -0.866),
    "D": (0.5, -0.866),
    "E": (0.5, 0.866),
    "F": (1.5, 0.866),
    "G": (2.5, 0.866),
    "H": (1.0, 0.0),
}
MAP_EDGES = (
    ("O", "A"),
    ("A", "B"),
    ("A", "G"),
    ("B", "C"),
    ("C", "D"),
    ("C", "H"),
    ("D", "E"),
    ("D", "H"),
    ("E", "F"),
    ("E", "H"),
    ("F", "G"),
    ("F", "H"),
)

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
region_colors = dict((node, None) for node in REGION_NODES)
uart_rx_buffer = ""
nav_previous_node = NAV_START_PREVIOUS_NODE
nav_current_node = NAV_START_CURRENT_NODE
nav_visited_mask = 0
nav_plan = None
nav_next_node = None
nav_turn_cmd = None


def ticks_ms():
    return int(time.time() * 1000)


def normalize_model_color(color_name):
    if color_name is None:
        return None

    color_text = str(color_name).upper()
    if color_text.startswith("R"):
        return "R"
    if color_text.startswith("G"):
        return "G"
    return None


def build_model_graph():
    graph = dict((node, []) for node in MODEL_NODES)
    for u, v in MAP_EDGES:
        ux, uy = MAP_POS[u]
        vx, vy = MAP_POS[v]
        weight = math.hypot(vx - ux, vy - uy) * MODEL_EDGE_CM
        graph[u].append((v, weight))
        graph[v].append((u, weight))
    return graph


MODEL_GRAPH = build_model_graph()


def visited_mask_with(node, visited_mask):
    if node in REGION_BIT:
        return visited_mask | REGION_BIT[node]
    return visited_mask


def make_visited_mask(nodes):
    visited_mask = 0
    for node in nodes:
        visited_mask = visited_mask_with(node, visited_mask)
    return visited_mask


def target_found_count():
    count = 0
    for node in REGION_NODES:
        if normalize_model_color(region_colors.get(node)) is not None:
            count += 1
    return count


def target_found_mask():
    found_mask = 0
    for node in REGION_NODES:
        if normalize_model_color(region_colors.get(node)) is not None:
            found_mask = visited_mask_with(node, found_mask)
    return found_mask


def turn_action(prev_node, curr_node, next_node):
    if prev_node is None:
        return "S"
    if prev_node == next_node:
        return "U"

    px, py = MAP_POS[prev_node]
    cx, cy = MAP_POS[curr_node]
    nx, ny = MAP_POS[next_node]
    in_x = cx - px
    in_y = cy - py
    out_x = nx - cx
    out_y = ny - cy
    cross = in_x * out_y - in_y * out_x

    if abs(cross) <= TURN_EPS:
        return "S"
    return "L" if cross > 0 else "R"


def is_legal_turn(prev_node, curr_node, next_node, colors):
    action = turn_action(prev_node, curr_node, next_node)
    if action == "U":
        return False

    color = normalize_model_color(colors.get(curr_node))
    if color == "R":
        return action == "R"
    if color == "G":
        return action == "L"
    return True


def route_actions(path, colors=None, previous_node=None):
    if colors is None:
        colors = region_colors

    actions = []
    if previous_node is not None and len(path) >= 2:
        curr_node = path[0]
        action = turn_action(previous_node, curr_node, path[1])
        actions.append(
            {
                "node": curr_node,
                "from": previous_node,
                "to": path[1],
                "action": action,
                "color": normalize_model_color(colors.get(curr_node)),
            }
        )

    for i in range(1, len(path) - 1):
        curr_node = path[i]
        action = turn_action(path[i - 1], curr_node, path[i + 1])
        actions.append(
            {
                "node": curr_node,
                "from": path[i - 1],
                "to": path[i + 1],
                "action": action,
                "color": normalize_model_color(colors.get(curr_node)),
            }
        )
    return actions


def reconstruct_route(parent, end_state):
    path = []
    state = end_state
    while state is not None:
        path.append(state[1])
        state = parent.get(state)
    path.reverse()
    return path


def plan_shortest_route(
    colors=None,
    start_node="O",
    previous_node=None,
    visited_mask=0,
    goal_mask=ALL_REGION_MASK,
    return_node=NAV_RETURN_NODE,
):
    if colors is None:
        colors = region_colors

    start_mask = visited_mask_with(start_node, visited_mask)
    start_state = (previous_node, start_node, start_mask)
    best_cost = {start_state: 0.0}
    parent = {start_state: None}
    queue = [(0.0, 0, start_state)]
    push_count = 1

    while queue:
        cost, _, state = heapq.heappop(queue)
        if cost > best_cost.get(state, float("inf")):
            continue

        prev_node, curr_node, visited_mask = state
        if (visited_mask & goal_mask) == goal_mask and curr_node == return_node:
            path = reconstruct_route(parent, state)
            return {
                "cost": cost,
                "path": path,
                "actions": route_actions(path, colors, previous_node),
            }

        for next_node, weight in MODEL_GRAPH[curr_node]:
            if not is_legal_turn(prev_node, curr_node, next_node, colors):
                continue

            next_mask = visited_mask_with(next_node, visited_mask)
            next_state = (curr_node, next_node, next_mask)
            next_cost = cost + weight
            if next_cost >= best_cost.get(next_state, float("inf")):
                continue

            best_cost[next_state] = next_cost
            parent[next_state] = state
            heapq.heappush(queue, (next_cost, push_count, next_state))
            push_count += 1

    return None


def update_region_color(region_node, color_name):
    if region_node not in region_colors:
        return None

    region_colors[region_node] = normalize_model_color(color_name)
    return update_navigation_plan()


def update_navigation_plan():
    global nav_plan, nav_next_node, nav_turn_cmd

    if target_found_count() >= TARGET_TOTAL:
        nav_plan = plan_shortest_route(
            region_colors,
            start_node=nav_current_node,
            previous_node=nav_previous_node,
            visited_mask=nav_visited_mask,
            goal_mask=0,
            return_node=NAV_RETURN_NODE,
        )
        if nav_plan is None or len(nav_plan["path"]) < 2:
            nav_next_node = None
            nav_turn_cmd = None
            return nav_plan

        nav_next_node = nav_plan["path"][1]
        nav_turn_cmd = turn_action(nav_previous_node, nav_current_node, nav_next_node)
        return nav_plan

    nav_plan = plan_shortest_route(
        region_colors,
        start_node=nav_current_node,
        previous_node=nav_previous_node,
        visited_mask=nav_visited_mask,
        return_node=NAV_RETURN_NODE,
    )
    if nav_plan is None or len(nav_plan["path"]) < 2:
        nav_next_node = None
        nav_turn_cmd = None
        return nav_plan

    nav_next_node = nav_plan["path"][1]
    nav_turn_cmd = turn_action(nav_previous_node, nav_current_node, nav_next_node)
    return nav_plan


def reset_navigation():
    global nav_previous_node, nav_current_node, nav_visited_mask

    nav_previous_node = NAV_START_PREVIOUS_NODE
    nav_current_node = NAV_START_CURRENT_NODE
    nav_visited_mask = visited_mask_with(nav_current_node, 0)
    update_navigation_plan()


def sync_navigation_state(previous_node, current_node, visited_nodes=None):
    global nav_previous_node, nav_current_node, nav_visited_mask

    if previous_node not in MODEL_NODES or current_node not in MODEL_NODES:
        return None

    nav_previous_node = previous_node
    nav_current_node = current_node
    nav_visited_mask = visited_mask_with(
        nav_current_node,
        make_visited_mask(visited_nodes or []),
    )
    return update_navigation_plan()


def mark_current_region_seen(color_name):
    global nav_visited_mask

    if nav_current_node in REGION_BIT:
        region_colors[nav_current_node] = normalize_model_color(color_name)
        nav_visited_mask = visited_mask_with(nav_current_node, nav_visited_mask)
    return update_navigation_plan()


def advance_navigation_to_next():
    global nav_previous_node, nav_current_node, nav_visited_mask

    if nav_next_node is None:
        update_navigation_plan()
    if nav_next_node is None:
        return None

    nav_previous_node, nav_current_node = nav_current_node, nav_next_node
    nav_visited_mask = visited_mask_with(nav_current_node, nav_visited_mask)
    return update_navigation_plan()


def nav_state_text():
    action = nav_turn_cmd if nav_turn_cmd is not None else "-"
    next_node = nav_next_node if nav_next_node is not None else "-"
    if target_found_count() >= TARGET_TOTAL:
        mode = "DONE" if nav_current_node == NAV_RETURN_NODE else "R"
    else:
        mode = "S"
    return "N:{} F:{}/{} M:{} A:{} T:{}".format(
        nav_current_node,
        target_found_count(),
        TARGET_TOTAL,
        mode,
        action,
        next_node,
    )


def build_result_message(shape_name, color_name):
    msg = "S:{},C:{}".format(shape_name, color_name)
    if UART_SEND_NAV_INFO:
        action = nav_turn_cmd if nav_turn_cmd is not None else "-"
        next_node = nav_next_node if nav_next_node is not None else "-"
        if target_found_count() >= TARGET_TOTAL:
            mode = "DONE" if nav_current_node == NAV_RETURN_NODE else "R"
        else:
            mode = "S"
        msg += ",N:{},F:{}/{},M:{},A:{},T:{}".format(
            nav_current_node,
            target_found_count(),
            TARGET_TOTAL,
            mode,
            action,
            next_node,
        )
    return msg


def handle_nav_command(line):
    command = line.strip().upper().replace(" ", "")
    if not command:
        return

    if command in ("ARR", "NAV:ARR"):
        advance_navigation_to_next()
        return

    if command in ("RESET", "NAV:RESET"):
        reset_navigation()
        return

    if command.startswith("SET:"):
        payload = command[4:]
    elif command.startswith("NAV:SET:"):
        payload = command[8:]
    else:
        return

    parts = payload.split(",")
    if len(parts) < 2:
        return

    visited_nodes = []
    if len(parts) >= 3 and parts[2]:
        visited_nodes = [node for node in parts[2].split("|") if node]
    sync_navigation_state(parts[0], parts[1], visited_nodes)


def poll_nav_commands():
    global uart_rx_buffer

    try:
        if not hasattr(serial1, "available") or serial1.available() <= 0:
            return
        data = serial1.read()
    except Exception:
        return

    if not data:
        return
    if isinstance(data, (bytes, bytearray)):
        text = data.decode("utf-8", "ignore")
    else:
        text = str(data)

    uart_rx_buffer += text
    while "\n" in uart_rx_buffer:
        line, uart_rx_buffer = uart_rx_buffer.split("\n", 1)
        handle_nav_command(line)


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
        uart_msg = build_result_message(shape_name, color_name)
        uart_burst_end_ms = now_ms + UART_BURST_MS
        last_send_ms = now_ms - UART_REPEAT_INTERVAL_MS

    if now_ms - last_send_ms < UART_REPEAT_INTERVAL_MS:
        return

    serial1.write_str(uart_msg + "\n")
    last_send_ms = now_ms


reset_navigation()


while not app.need_exit():
    poll_nav_commands()

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
        mark_current_region_seen(color_name)
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

    cv2.putText(
        img_cv,
        nav_state_text(),
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )

    disp.show(img)
