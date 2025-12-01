import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import openai
import json
import urllib.parse
import math
from PIL import Image, ImageDraw, ImageFont

# =========================================
# --- 1. 配置区域 ---
# =========================================
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# --- 2. 核心工具函数 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a);
    b = np.array(b);
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return int(angle)


def calculate_vertical_angle(a, b):
    a = np.array(a);
    b = np.array(b)
    radians = np.arctan2(a[0] - b[0], a[1] - b[1])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180: angle = 360 - angle
    return int(180 - angle)


# 🌟 中文支持
def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=20):
    try:
        if (isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("msyhbd.ttc", text_size)
        except:
            font = ImageFont.load_default()
        draw.text(position, text, fill=text_color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    except:
        return img


# 🌟 弯曲箭头绘制
def draw_curved_arrow(img, center, start_pt, end_pt, color, thickness=3):
    radius = int(np.linalg.norm(np.array(start_pt) - np.array(center)))
    angle_start = math.degrees(math.atan2(start_pt[1] - center[1], start_pt[0] - center[0]))
    angle_end = math.degrees(math.atan2(end_pt[1] - center[1], end_pt[0] - center[0]))
    if angle_start < 0: angle_start += 360
    if angle_end < 0: angle_end += 360
    if abs(angle_start - angle_end) > 180:
        if angle_start > angle_end:
            angle_end += 360
        else:
            angle_start += 360
    cv2.ellipse(img, tuple(center), (radius, radius), 0, angle_start, angle_end, color, thickness, cv2.LINE_AA)
    tip_angle = math.radians(angle_end)
    tip_x = int(center[0] + radius * math.cos(tip_angle))
    tip_y = int(center[1] + radius * math.sin(tip_angle))
    cv2.circle(img, (tip_x, tip_y), thickness + 3, color, -1)


# 🌟 虚线绘制
def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=10):
    dist = np.linalg.norm(pt1 - pt2)
    dashes = int(dist / dash_len)
    for i in range(dashes):
        start = pt1 + (pt2 - pt1) * (i / dashes)
        end = pt1 + (pt2 - pt1) * ((i + 0.5) / dashes)
        cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)


# 🌟 绘图引擎 (核心升级)
def draw_values_on_body(image, angles, p_coords, mode="basic"):
    h, w, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8 if w < 1000 else 1.2
    thick = 2 if w < 1000 else 3

    def to_pix(p):
        return np.multiply(p, [w, h]).astype(int)

    def to_tuple(p):
        return tuple(to_pix(p))

    def draw_txt(txt, pos, color, offset=(0, 0)):
        p = (pos[0] + offset[0], pos[1] + offset[1])
        cv2.putText(image, txt, p, font, font_scale * 0.8, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(image, txt, p, font, font_scale * 0.8, color, thick, cv2.LINE_AA)

    hip_pt = to_pix(p_coords['hip'])
    knee_pt = to_pix(p_coords['knee'])
    shoulder_pt = to_pix(p_coords['shoulder'])
    ankle_pt = to_pix(p_coords['ankle'])

    # --- 基础层：画原始骨骼线和数字 ---
    # 1. 画原始骨骼线 (恢复红/蓝实线)
    cv2.line(image, tuple(hip_pt), tuple(shoulder_pt), (0, 0, 255), 3, cv2.LINE_AA)  # 躯干红线
    cv2.line(image, tuple(hip_pt), tuple(knee_pt), (255, 0, 0), 3, cv2.LINE_AA)  # 大腿蓝线
    cv2.line(image, tuple(knee_pt), tuple(ankle_pt), (255, 0, 0), 3, cv2.LINE_AA)  # 小腿蓝线

    # 2. 画数字
    draw_txt(f"{angles['knee']}", to_tuple(p_coords['knee']), (0, 255, 255), offset=(10, 0))
    draw_txt(f"{angles['hip']}", to_tuple(p_coords['hip']), (0, 255, 0), offset=(10, 0))
    draw_txt(f"L:{angles['trunk']}", to_tuple(p_coords['shoulder']), (0, 0, 255), offset=(0, -20))

    # --- 纠正层：画理想绿线、引导虚线、箭头和中文 ---
    if mode == "correction":
        facing_right = knee_pt[0] > hip_pt[0]
        direction = 1 if facing_right else -1

        # 1. 躯干纠正 (多画一条白色虚线引导)
        if angles['trunk'] > 40:
            trunk_len = np.linalg.norm(shoulder_pt - hip_pt)
            ideal_x = hip_pt[0] + direction * trunk_len * math.sin(math.radians(30))
            ideal_y = hip_pt[1] - trunk_len * math.cos(math.radians(30))
            ideal_pt = np.array([ideal_x, ideal_y]).astype(int)

            # A. 画理想绿线 (实线)
            cv2.line(image, tuple(hip_pt), tuple(ideal_pt), (0, 255, 0), 4, cv2.LINE_AA)
            # B. 画引导虚线 (从原始肩膀连到理想肩膀)
            draw_dashed_line(image, shoulder_pt, ideal_pt, (255, 255, 255), thickness=2)
            # C. 画弯曲箭头
            draw_curved_arrow(image, hip_pt, shoulder_pt, ideal_pt, (255, 255, 255), 5)

            t_x = shoulder_pt[0] - 120 if facing_right else shoulder_pt[0] + 20
            image = cv2_add_chinese_text(image, "躯干后挺", (int(t_x), int(shoulder_pt[1] - 40)), (0, 255, 0), 25)

        # 2. 深度纠正 (多画一条垂直参考线)
        if angles['knee'] > 95:
            # A. 画理想水平线
            ideal_knee_x = hip_pt[0] + direction * (np.linalg.norm(knee_pt - hip_pt) * 1.2)
            cv2.line(image, tuple(hip_pt), (int(ideal_knee_x), int(hip_pt[1])), (0, 255, 0), 3)
            # B. 画当前膝盖的垂直参考线
            draw_dashed_line(image, knee_pt, (knee_pt[0], hip_pt[1]), (255, 255, 0), thickness=2)

            image = cv2_add_chinese_text(image, "继续下蹲", (int(ideal_knee_x), int(hip_pt[1] - 30)), (0, 255, 255), 25)

        # 3. 重心线
        start_pt = shoulder_pt
        draw_dashed_line(image, start_pt, (start_pt[0], h), (0, 0, 255), thickness=2, dash_len=15)
        image = cv2_add_chinese_text(image, "重心垂线", (start_pt[0] - 40, h - 30), (0, 0, 255), 20)

    return image


def make_search_link(exercise_name):
    query = urllib.parse.quote(str(exercise_name) + " 动作教学")
    return f"https://www.bilibili.com/search?keyword={query}"


def get_references(height_cm):
    trunk_limit = 50 if height_cm > 180 else (35 if height_cm < 165 else 45)
    return {"knee": "< 90°", "hip": "40°~60°", "trunk": f"<{trunk_limit}°", "ankle": "< 75°",
            "trunk_limit": trunk_limit}


def safe_exercise_parser(ex_data):
    if isinstance(ex_data, str): return {"name": ex_data, "sets_reps": "3组 x 10次", "load": "自重",
                                         "note": "保持标准姿态"}
    return ex_data


# --- 3. 分析引擎 (保持不变) ---
def analyze_with_ai_json(api_key, biomech_data, one_rm, height, refs):
    if not api_key: return None
    client = openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    system_prompt = f"""
    你是一位世界级力量举教练。请根据受试者数据（1RM {one_rm}kg, 身高{height}cm）生成【极度详细】的训练处方。

    【输出要求 - 拒绝简略】:
    1. **T1 主项**: 必须基于 1RM 计算具体负荷 (kg)。例如: "85kg (85% 1RM)"。
    2. **动作细节**: T2/T3/T4 每个动作都必须包含具体的组数、次数、负荷建议（如RPE或自重）。
    3. **针对性**: 热身和辅助动作必须直接解决用户的生物力学弱点（如踝受限）。
    4. **语言**: 全程中文。

    返回 JSON:
    {{
        "diagnosis": {{ "summary": "...", "detailed": [{{ "part": "...", "status": "...", "issue": "..." }}] }},
        "prescription": {{
            "warmup": {{ "focus": "...", "exercises": [{{ "name": "...", "sets_reps": "...", "note": "..." }}] }},
            "t1": {{ "action": "...", "decision": "...", "plan": "...", "load": "...", "reason": "..." }},
            "t2": {{ "focus": "...", "exercises": [{{ "name": "...", "sets_reps": "...", "load": "...", "note": "..." }}] }},
            "t3": {{ "focus": "...", "exercises": [{{ "name": "...", "sets_reps": "...", "load": "...", "note": "..." }}] }},
            "t4": {{ "exercises": [{{ "name": "...", "sets_reps": "...", "load": "...", "note": "..." }}] }}
        }}
    }}
    """
    user_prompt = str(biomech_data)
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.3, response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.replace("```json", "").replace("```", "")
        return json.loads(content)
    except:
        return None


# --- 4. 主程序 (保持不变) ---
def main():
    st.set_page_config(page_title="运动动作深度分析系统", layout="wide")
    st.title = "🧬 运动动作深度分析系统 (V30.0 完美交付版)"

    st.sidebar.header("🔑 系统密钥")
    api_key = st.sidebar.text_input("授权密钥", type="password")
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("上传视频素材", type=["mp4", "mov", "avi"])
    if st.sidebar.button("🔄 重置"): st.rerun()

    col_video, col_data = st.columns([2, 3])

    with col_data:
        st.subheader("📊 实时数据监测")
        r1, r2, r3, r4 = st.columns(4)
        p_knee = r1.empty();
        p_hip = r2.empty();
        p_trunk = r3.empty();
        p_ankle = r4.empty()
        st.markdown("---")
        status_text = st.empty()
        report_container = st.container()

    if uploaded_file is not None:
        with col_video:
            st.markdown("### 📋 受试者档案")
            c_h, c_w = st.columns(2)
            height = c_h.number_input("身高 (cm)", 140, 220, 175)
            one_rm = c_w.number_input("深蹲 1RM (kg)", 1, 500, 100)
            refs = get_references(height)
            st.caption(f"📏 标准躯干范围: **{refs['trunk']}** (基于身高修正)")

        tfile = tempfile.NamedTemporaryFile(delete=False);
        tfile.write(uploaded_file.read())
        vf = cv2.VideoCapture(tfile.name)

        ranges = {k: {"min": 360, "max": 0} for k in ["knee", "hip", "trunk", "ankle"]}
        min_knee_rec = 360;
        best_stats = {};
        best_frame_basic = None;
        best_frame_correction = None
        frame_count = 0

        with col_video:
            st_vid = st.empty()
            st_correction = st.empty()

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                frame_count += 1
                if frame.shape[0] < frame.shape[1]: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    lm = results.pose_landmarks.landmark
                    p_c = {
                        'shoulder': [lm[11].x, lm[11].y], 'hip': [lm[23].x, lm[23].y],
                        'knee': [lm[25].x, lm[25].y], 'ankle': [lm[27].x, lm[27].y], 'foot': [lm[31].x, lm[31].y]
                    }
                    ang = {
                        "knee": calculate_angle(p_c['hip'], p_c['knee'], p_c['ankle']),
                        "hip": calculate_angle(p_c['shoulder'], p_c['hip'], p_c['knee']),
                        "trunk": calculate_vertical_angle(p_c['shoulder'], p_c['hip']),
                        "ankle": calculate_angle(p_c['knee'], p_c['ankle'], p_c['foot'])
                    }
                    for k, v in ang.items():
                        if v < ranges[k]["min"]: ranges[k]["min"] = v
                        if v > ranges[k]["max"]: ranges[k]["max"] = v

                    img_display = draw_values_on_body(img_bgr.copy(), ang, p_c, mode="basic")

                    if ang['knee'] < min_knee_rec:
                        min_knee_rec = ang['knee']
                        best_frame_basic = img_display.copy()
                        # 🌟 关键修改：这里生成的纠正图，会包含原线、绿线和白虚线
                        best_frame_correction = draw_values_on_body(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), ang, p_c,
                                                                    mode="correction")
                        best_stats = ang.copy()
                        bottom_frame_index = frame_count

                    p_knee.metric("🦵 膝角", f"{ranges['knee']['min']}~{ranges['knee']['max']}", f"{ang['knee']}°")
                    p_hip.metric("📐 髋角", f"{ranges['hip']['min']}~{ranges['hip']['max']}", f"{ang['hip']}°")
                    p_trunk.metric("🧍 躯干", f"{ranges['trunk']['min']}~{ranges['trunk']['max']}", f"{ang['trunk']}°")
                    p_ankle.metric("🦶 踝角", f"{ranges['ankle']['min']}~{ranges['ankle']['max']}", f"{ang['ankle']}°")
                    status_text.info(f"▶️ 采样中... Frame: {frame_count}")
                st_vid.image(img_display, channels="BGR", use_container_width=True)

        status_text.success("✅ 分析完成，正在生成专家级处方...")
        descent = bottom_frame_index if 'bottom_frame_index' in locals() else frame_count // 2
        ascent = frame_count - descent
        tempo = descent / ascent if ascent > 0 else 0

        if best_frame_basic is not None:
            st_vid.image(best_frame_basic, caption=f"📸 动作最低点定格 (膝角: {min_knee_rec}°)", channels="BGR",
                         use_container_width=True)
            if best_frame_correction is not None:
                st_correction.image(best_frame_correction,
                                    caption="🧠 AI 视觉纠正教学 (红/蓝线为原始, 绿色为理想, 白色虚线为引导)",
                                    channels="BGR", use_container_width=True)

            with report_container:
                st.subheader("🏆 核心指标监测")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📉 膝角", f"{best_stats['knee']}°");
                c1.caption(f"标准: {refs['knee']}")
                c2.metric("📐 髋角", f"{best_stats['hip']}°");
                c2.caption(f"标准: {refs['hip']}")
                t_delta = "正常" if best_stats['trunk'] <= refs['trunk_limit'] else "过大"
                c3.metric("🧍 躯干", f"{best_stats['trunk']}°", delta=t_delta,
                          delta_color="normal" if t_delta == "正常" else "inverse");
                c3.caption(f"标准: {refs['trunk']}")
                c4.metric("🦶 踝角", f"{best_stats['ankle']}°");
                c4.caption(f"标准: {refs['ankle']}")
                st.markdown("---")

                payload = {"static": best_stats, "dynamic": {"tempo_ratio": tempo}}
                ai_result = analyze_with_ai_json(api_key, payload, one_rm, height, refs)

                if ai_result:
                    st.subheader("📋 深度诊断报告")
                    st.info(f"💡 **综合点评:** {ai_result.get('diagnosis', {}).get('summary')}")
                    for item in ai_result.get('diagnosis', {}).get('detailed', []):
                        st.markdown(f"- **{item.get('part')}**: `{item.get('status')}` → {item.get('issue')}")
                    st.markdown("---")

                    st.subheader("💊 运动干预处方")
                    pres = ai_result.get('prescription', {})

                    warmup = pres.get('warmup', {})
                    st.markdown(f"**🔥 T0 热身与激活 (针对: {warmup.get('focus')})**")
                    if 'exercises' in warmup:
                        for raw_ex in warmup['exercises']:
                            ex = safe_exercise_parser(raw_ex)
                            st.markdown(
                                f"- [📺 {ex.get('name')}]({make_search_link(ex.get('name'))}) | `{ex.get('sets_reps', '-')}` | _{ex.get('note', '')}_")
                    st.divider()

                    t1 = pres.get('t1', {})
                    st.markdown(f"#### 🏋️‍♂️ T1 主项: {t1.get('action')}")
                    color = "red" if "退阶" in t1.get('decision', '') or "减重" in t1.get('decision', '') else "green"
                    st.markdown(f":{color}[**决策: {t1.get('decision')}**]")
                    t1_cols = st.columns(3)
                    t1_cols[0].metric("计划", t1.get('plan'))
                    t1_cols[1].metric("负荷", t1.get('load'))
                    t1_cols[2].caption(f"💡 {t1.get('reason')}")
                    st.divider()

                    t2 = pres.get('t2', {})
                    st.subheader("🛡️ T2 辅助补强训练")
                    st.caption(f"🎯 重点: {t2.get('focus')}")
                    if 'exercises' in t2:
                        for raw_ex in t2['exercises']:
                            ex = safe_exercise_parser(raw_ex)
                            with st.container():
                                c1, c2 = st.columns([3, 2])
                                c1.markdown(f"**{ex.get('name')}** [📺 演示]({make_search_link(ex.get('name'))})")
                                c1.caption(f"📝 {ex.get('note')}")
                                c2.markdown(f"`{ex.get('sets_reps')}` | ⚖️ {ex.get('load')}")
                            st.markdown("")
                    st.divider()

                    t3 = pres.get('t3', {})
                    st.subheader("🧘 T3 修正与稳定性训练")
                    st.caption(f"🎯 重点: {t3.get('focus')}")
                    if 'exercises' in t3:
                        for raw_ex in t3['exercises']:
                            ex = safe_exercise_parser(raw_ex)
                            with st.container():
                                c1, c2 = st.columns([3, 2])
                                c1.markdown(f"**{ex.get('name')}** [📺 演示]({make_search_link(ex.get('name'))})")
                                c1.caption(f"📝 {ex.get('note')}")
                                c2.markdown(f"`{ex.get('sets_reps')}` | ⚖️ {ex.get('load')}")
                            st.markdown("")

                    st.divider()

                    t4 = pres.get('t4', {})
                    st.subheader("🦾 T4 康复与核心辅助")
                    if 'exercises' in t4:
                        for raw_ex in t4['exercises']:
                            ex = safe_exercise_parser(raw_ex)
                            st.markdown(
                                f"- **{ex.get('name')}** | `{ex.get('sets_reps')}` | ⚖️ {ex.get('load', '自重')} | _{ex.get('note')}_")

                else:
                    st.error("⚠️ 无法生成处方，请检查密钥。")


if __name__ == '__main__':
    main()
