import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from modules.gpu_processing import gpu_cvt_color, gpu_resize, gpu_flip
from PIL import Image, ImageOps
import time
import json
import queue
import threading
import numpy as np
import requests
import tempfile
import modules.globals
import modules.metadata
from modules.face_analyser import (
    get_one_face,
    get_many_faces,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    add_blank_map,
    has_valid_map,
    simplify_maps,
)
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)
from modules.video_capture import VideoCapturer
from modules.gettext import LanguageManager
from modules.ui_tooltip import ToolTip
from modules import globals
import platform

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

# --- Tk 9.0 compatibility patch ---
# In Tk 9.0, Menu.index("end") returns "" instead of raising TclError
# when the menu is empty. CustomTkinter's CTkOptionMenu doesn't handle
# this, causing crashes. This patch adds the missing guard.
try:
    from customtkinter.windows.widgets.core_widget_classes import DropdownMenu as _DropdownMenu

    _original_add_menu_commands = _DropdownMenu._add_menu_commands

    def _patched_add_menu_commands(self, *args, **kwargs):
        try:
            end_index = self._menu.index("end")
            if end_index == "" or end_index is None:
                return
        except Exception:
            pass
        _original_add_menu_commands(self, *args, **kwargs)

    _DropdownMenu._add_menu_commands = _patched_add_menu_commands
except (ImportError, AttributeError):
    pass  # CustomTkinter version doesn't have this class path
# --- End Tk 9.0 patch ---

ROOT = None
POPUP = None
POPUP_LIVE = None
ROOT_HEIGHT = 800
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 960
PREVIEW_DEFAULT_HEIGHT = 540

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = (740,)
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = (890,)
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_MAX_HEIGHT = 100
MAPPER_PREVIEW_MAX_WIDTH = 100

DEFAULT_BUTTON_WIDTH = 200
DEFAULT_BUTTON_HEIGHT = 40

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

_ = None
preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
popup_status_label = None
popup_status_label_live = None
source_label_dict = {}
source_label_dict_live = {}
target_label_dict_live = {}

img_ft, vid_ft = modules.globals.file_types


def init(start: Callable[[], None], destroy: Callable[[], None], lang: str) -> ctk.CTk:
    global ROOT, PREVIEW, _

    lang_manager = LanguageManager(lang)
    _ = lang_manager._
    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def save_switch_states():
    switch_states = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
    }
    with open("switch_states.json", "w") as f:
        json.dump(switch_states, f)


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            switch_states = json.load(f)
        modules.globals.keep_fps = switch_states.get("keep_fps", True)
        modules.globals.keep_audio = switch_states.get("keep_audio", True)
        modules.globals.keep_frames = switch_states.get("keep_frames", False)
        modules.globals.many_faces = switch_states.get("many_faces", False)
        modules.globals.map_faces = switch_states.get("map_faces", False)
        modules.globals.poisson_blend = switch_states.get("poisson_blend", False)
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask_size = switch_states.get("mouth_mask_size", 0.0)
        # mouth_mask is driven by the slider: on if size > 0, off if 0
        modules.globals.mouth_mask = modules.globals.mouth_mask_size > 0
        modules.globals.show_mouth_mask_box = False  # always start hidden
    except FileNotFoundError:
        # If the file doesn't exist, use default values
        pass


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, show_fps_switch

    load_switch_states()

    try:
        ctk.deactivate_automatic_dpi_awareness()
    except Exception:
        pass

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.geometry("780x880")
    root.minsize(740, 820)
    root.title(f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition} (ภาษาไทย)")
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    # --- Header Bar ---
    header_frame = ctk.CTkFrame(root, corner_radius=10, fg_color="#1E1E2E")
    header_frame.pack(fill="x", padx=15, pady=(15, 10))

    title_label = ctk.CTkLabel(
        header_frame,
        text="Deep-Live-Cam 2.1",
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color="#3B82F6",
    )
    title_label.pack(side="left", padx=15, pady=10)

    subtitle_label = ctk.CTkLabel(
        header_frame,
        text="ระบบสลับใบหน้าและเปลี่ยนหน้าคนในวิดีโอ",
        font=ctk.CTkFont(size=13),
        text_color="#9CA3AF",
    )
    subtitle_label.pack(side="left", padx=5, pady=10)

    def toggle_theme(choice):
        mode_map = {"ธีมมืด (Dark)": "dark", "ธีมสว่าง (Light)": "light", "ตามระบบ (System)": "system"}
        ctk.set_appearance_mode(mode_map.get(choice, "dark"))

    theme_menu = ctk.CTkOptionMenu(
        header_frame,
        values=["ธีมมืด (Dark)", "ธีมสว่าง (Light)", "ตามระบบ (System)"],
        command=toggle_theme,
        width=130,
        height=28,
    )
    theme_menu.pack(side="right", padx=15, pady=10)
    theme_menu.set("ธีมมืด (Dark)")

    # --- Main Scrollable Container ---
    main_scroll = ctk.CTkScrollableFrame(root, fg_color="transparent")
    main_scroll.pack(fill="both", expand=True, padx=15, pady=5)

    # 2-Column Grid inside main_scroll
    main_scroll.columnconfigure(0, weight=1)
    main_scroll.columnconfigure(1, weight=1)

    # ==================== LEFT COLUMN: Media Preview Cards ====================
    left_frame = ctk.CTkFrame(main_scroll, fg_color="transparent")
    left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)

    # 1. Source Face Card
    source_card = ctk.CTkFrame(left_frame, corner_radius=10)
    source_card.pack(fill="x", pady=(0, 12))

    source_title = ctk.CTkLabel(
        source_card, text="👤 รูปภาพใบหน้าต้นฉบับ", font=ctk.CTkFont(size=14, weight="bold")
    )
    source_title.pack(anchor="w", padx=15, pady=(10, 5))

    source_preview_frame = ctk.CTkFrame(source_card, height=180, fg_color="#181825", corner_radius=8)
    source_preview_frame.pack(fill="x", padx=15, pady=5)
    source_preview_frame.pack_propagate(False)

    source_label = ctk.CTkLabel(source_preview_frame, text="คลิกปุ่ม 'เลือกรูปใบหน้า' ด้านล่าง", text_color="#6B7280", font=ctk.CTkFont(size=12))
    source_label.pack(fill="both", expand=True, padx=5, pady=5)

    btn_frame_1 = ctk.CTkFrame(source_card, fg_color="transparent")
    btn_frame_1.pack(fill="x", padx=15, pady=(5, 12))

    select_face_button = ctk.CTkButton(
        btn_frame_1,
        text="📷 เลือกรูปใบหน้า",
        font=ctk.CTkFont(size=13),
        cursor="hand2",
        command=lambda: select_source_path(),
        fg_color="#2563EB",
        hover_color="#1D4ED8",
    )
    select_face_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
    ToolTip(select_face_button, "เลือกรูปภาพใบหน้าของคนที่ต้องการนำไปแปะสลับ")

    random_face_button = ctk.CTkButton(
        btn_frame_1,
        text="🔄 สุ่มรูป",
        font=ctk.CTkFont(size=12),
        cursor="hand2",
        width=75,
        command=lambda: fetch_random_face(),
        fg_color="#374151",
        hover_color="#4B5563",
    )
    random_face_button.pack(side="right")
    ToolTip(random_face_button, "ดาวน์โหลดรูปสุ่มใบหน้าจากอินเทอร์เน็ต")

    # 2. Target Media Card
    target_card = ctk.CTkFrame(left_frame, corner_radius=10)
    target_card.pack(fill="x", pady=0)

    target_title = ctk.CTkLabel(
        target_card, text="🎬 วิดีโอ หรือ รูปภาพเป้าหมาย", font=ctk.CTkFont(size=14, weight="bold")
    )
    target_title.pack(anchor="w", padx=15, pady=(10, 5))

    target_preview_frame = ctk.CTkFrame(target_card, height=180, fg_color="#181825", corner_radius=8)
    target_preview_frame.pack(fill="x", padx=15, pady=5)
    target_preview_frame.pack_propagate(False)

    target_label = ctk.CTkLabel(target_preview_frame, text="คลิกปุ่ม 'เลือกวิดีโอ/รูปภาพ' ด้านล่าง", text_color="#6B7280", font=ctk.CTkFont(size=12))
    target_label.pack(fill="both", expand=True, padx=5, pady=5)

    btn_frame_2 = ctk.CTkFrame(target_card, fg_color="transparent")
    btn_frame_2.pack(fill="x", padx=15, pady=(5, 12))

    select_target_button = ctk.CTkButton(
        btn_frame_2,
        text="🎞 เลือกวิดีโอ/รูปภาพ",
        font=ctk.CTkFont(size=13),
        cursor="hand2",
        command=lambda: select_target_path(),
        fg_color="#2563EB",
        hover_color="#1D4ED8",
    )
    select_target_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
    ToolTip(select_target_button, "เลือกไฟล์วิดีโอหรือรูปภาพที่ต้องการเปลี่ยนหน้า")

    swap_faces_button = ctk.CTkButton(
        btn_frame_2,
        text="↔ สลับรูป",
        font=ctk.CTkFont(size=12),
        cursor="hand2",
        width=75,
        command=lambda: swap_faces_paths(),
        fg_color="#374151",
        hover_color="#4B5563",
    )
    swap_faces_button.pack(side="right")
    ToolTip(swap_faces_button, "สลับตำแหน่งระหว่างรูปต้นฉบับกับรูปเป้าหมาย")

    # ==================== RIGHT COLUMN: Controls & Settings Cards ====================
    right_frame = ctk.CTkFrame(main_scroll, fg_color="transparent")
    right_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)

    # 1. AI Models & Enhancer Card
    enhancer_card = ctk.CTkFrame(right_frame, corner_radius=10)
    enhancer_card.pack(fill="x", pady=(0, 12))

    enhancer_title = ctk.CTkLabel(
        enhancer_card, text="✨ ปรับแต่งคุณภาพและความคมชัด", font=ctk.CTkFont(size=14, weight="bold")
    )
    enhancer_title.pack(anchor="w", padx=15, pady=(10, 5))

    # Enhancer Dropdown
    enh_sub_frame = ctk.CTkFrame(enhancer_card, fg_color="transparent")
    enh_sub_frame.pack(fill="x", padx=15, pady=4)
    ctk.CTkLabel(enh_sub_frame, text="โมเดลปรับความคมชัด:", font=ctk.CTkFont(size=12)).pack(side="left")

    enhancer_options_display = ["ไม่ปรับแต่ง (None)", "GFPGAN", "GPEN-512", "GPEN-256"]
    enhancer_map_to_key = {
        "ไม่ปรับแต่ง (None)": None,
        "GFPGAN": "face_enhancer",
        "GPEN-512": "face_enhancer_gpen512",
        "GPEN-256": "face_enhancer_gpen256",
    }
    initial_enhancer_display = "ไม่ปรับแต่ง (None)"
    if modules.globals.fp_ui.get("face_enhancer", False):
        initial_enhancer_display = "GFPGAN"
    elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
        initial_enhancer_display = "GPEN-512"
    elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
        initial_enhancer_display = "GPEN-256"

    enhancer_variable = ctk.StringVar(value=initial_enhancer_display)

    def on_enhancer_change(choice: str):
        for key in ["face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"]:
            update_tumbler(key, False)
        selected_key = enhancer_map_to_key.get(choice)
        if selected_key:
            update_tumbler(selected_key, True)
        save_switch_states()

    enhancer_dropdown = ctk.CTkOptionMenu(
        enh_sub_frame,
        variable=enhancer_variable,
        values=enhancer_options_display,
        command=on_enhancer_change,
        width=140,
        height=26,
    )
    enhancer_dropdown.pack(side="right")
    ToolTip(enhancer_dropdown, "เลือกโมเดลช่วยปรับความคมชัดของใบหน้าผลลัพธ์")

    # Transparency Slider
    trans_frame = ctk.CTkFrame(enhancer_card, fg_color="transparent")
    trans_frame.pack(fill="x", padx=15, pady=4)
    ctk.CTkLabel(trans_frame, text="ความโปร่งใส (Opacity):", font=ctk.CTkFont(size=12)).pack(side="left")

    transparency_var = ctk.DoubleVar(value=1.0)

    def on_transparency_change(value: float):
        val = float(value)
        modules.globals.opacity = val
        percentage = int(val * 100)
        if percentage == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("ปรับความโปร่งใสเป็น 0% - ปิดการสลับใบหน้า")
        elif percentage == 100:
            modules.globals.face_swapper_enabled = True
            update_status("ปรับความโปร่งใสเป็น 100% (สลับหน้าเต็มรูปแบบ)")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"ปรับความโปร่งใสเป็น {percentage}%")

    transparency_slider = ctk.CTkSlider(
        trans_frame, from_=0.0, to=1.0, variable=transparency_var, command=on_transparency_change, width=130, height=16
    )
    transparency_slider.pack(side="right")
    ToolTip(transparency_slider, "ปรับระดับความเนียนในการผสมระหว่างหน้าเดิมกับหน้าใหม่")

    # Sharpness Slider
    sharp_frame = ctk.CTkFrame(enhancer_card, fg_color="transparent")
    sharp_frame.pack(fill="x", padx=15, pady=4)
    ctk.CTkLabel(sharp_frame, text="ความคมชัด (Sharpness):", font=ctk.CTkFont(size=12)).pack(side="left")

    sharpness_var = ctk.DoubleVar(value=0.0)

    def on_sharpness_change(value: float):
        modules.globals.sharpness = float(value)
        update_status(f"ปรับความคมชัดเป็น {value:.1f}")

    sharpness_slider = ctk.CTkSlider(
        sharp_frame, from_=0.0, to=5.0, variable=sharpness_var, command=on_sharpness_change, width=130, height=16
    )
    sharpness_slider.pack(side="right")
    ToolTip(sharpness_slider, "เร่งความคมชัดภาพส่วนใบหน้า")

    # Mouth Mask Size Slider
    mouth_frame = ctk.CTkFrame(enhancer_card, fg_color="transparent")
    mouth_frame.pack(fill="x", padx=15, pady=(4, 10))
    ctk.CTkLabel(mouth_frame, text="ขอบเขตปาก (Mouth Mask):", font=ctk.CTkFont(size=12)).pack(side="left")

    mouth_mask_size_var = ctk.DoubleVar(value=modules.globals.mouth_mask_size)
    mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)

    def on_mouth_mask_size_change(value: float):
        val = float(value)
        modules.globals.mouth_mask_size = val
        if val > 0:
            modules.globals.mouth_mask = True
            mouth_mask_var.set(True)
        else:
            modules.globals.mouth_mask = False
            mouth_mask_var.set(False)
            modules.globals.show_mouth_mask_box = False

    def on_mouth_mask_slider_release(event):
        modules.globals.show_mouth_mask_box = False

    def on_mouth_mask_slider_press(event):
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    mouth_mask_size_slider = ctk.CTkSlider(
        mouth_frame, from_=0.0, to=100.0, variable=mouth_mask_size_var, command=on_mouth_mask_size_change, width=130, height=16
    )
    mouth_mask_size_slider.pack(side="right")
    mouth_mask_size_slider.bind("<ButtonPress-1>", on_mouth_mask_slider_press)
    mouth_mask_size_slider.bind("<ButtonRelease-1>", on_mouth_mask_slider_release)
    ToolTip(mouth_mask_size_slider, "0 = ใช้ปากสลับหน้า, 100 = เปิดเผยริมฝีปากและคางเดิมเพื่อความสมจริง")

    # 2. Pipeline Switches Card
    switches_card = ctk.CTkFrame(right_frame, corner_radius=10)
    switches_card.pack(fill="x", pady=0)

    switches_title = ctk.CTkLabel(
        switches_card, text="⚙️ ตัวเลือกการประมวลผล", font=ctk.CTkFont(size=14, weight="bold")
    )
    switches_title.pack(anchor="w", padx=15, pady=(10, 5))

    sw_grid = ctk.CTkFrame(switches_card, fg_color="transparent")
    sw_grid.pack(fill="x", padx=15, pady=(0, 10))
    sw_grid.columnconfigure(0, weight=1)
    sw_grid.columnconfigure(1, weight=1)

    keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(
        sw_grid, text="คงค่า FPS เดิม", variable=keep_fps_value, cursor="hand2", font=ctk.CTkFont(size=12),
        command=lambda: (setattr(modules.globals, "keep_fps", keep_fps_value.get()), save_switch_states())
    )
    keep_fps_checkbox.grid(row=0, column=0, sticky="w", pady=4)
    ToolTip(keep_fps_checkbox, "วิดีโอผลลัพธ์จะคงความลื่นไหลเฟรมเรตเท่าเดิม")

    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(
        sw_grid, text="รักษาเสียงเดิม", variable=keep_audio_value, cursor="hand2", font=ctk.CTkFont(size=12),
        command=lambda: (setattr(modules.globals, "keep_audio", keep_audio_value.get()), save_switch_states())
    )
    keep_audio_switch.grid(row=0, column=1, sticky="w", pady=4)
    ToolTip(keep_audio_switch, "คัดลอกเสียงจากวิดีโอต้นฉบับมาใส่ในไฟล์ผลลัพธ์")

    keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(
        sw_grid, text="เก็บไฟล์เฟรมชั่วคราว", variable=keep_frames_value, cursor="hand2", font=ctk.CTkFont(size=12),
        command=lambda: (setattr(modules.globals, "keep_frames", keep_frames_value.get()), save_switch_states())
    )
    keep_frames_switch.grid(row=1, column=0, sticky="w", pady=4)
    ToolTip(keep_frames_switch, "เก็บไฟล์รูปภาพเฟรมในดิสก์ไว้หลังจากประมวลผลเสร็จ")

    many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(
        sw_grid, text="สลับทุกใบหน้า", variable=many_faces_value, cursor="hand2", font=ctk.CTkFont(size=12),
        command=lambda: (setattr(modules.globals, "many_faces", many_faces_value.get()), save_switch_states())
    )
    many_faces_switch.grid(row=1, column=1, sticky="w", pady=4)
    ToolTip(many_faces_switch, "สลับหน้าทุกคนที่ตรวจพบในเฟรมภาพ")

    map_faces = ctk.BooleanVar(value=modules.globals.map_faces)
    map_faces_switch = ctk.CTkSwitch(
        sw_grid, text="จับคู่ใบหน้ากำหนดเอง", variable=map_faces, cursor="hand2", font=ctk.CTkFont(size=12),
        command=lambda: (
            setattr(modules.globals, "map_faces", map_faces.get()),
            save_switch_states(),
            close_mapper_window() if not map_faces.get() else None
        )
    )
    map_faces_switch.grid(row=2, column=0, sticky="w", pady=4)
    ToolTip(map_faces_switch, "เปิดหน้าต่างจับคู่ว่าคนไหนสลับกับใบหน้าคนไหน")

    poisson_blend_value = ctk.BooleanVar(value=modules.globals.poisson_blend)
    poisson_blend_switch = ctk.CTkSwitch(
        sw_grid, text="ผสมขอบเนียน (Poisson)", variable=poisson_blend_value, cursor="hand2", font=ctk.CTkFont(size=12),
        command=lambda: (setattr(modules.globals, "poisson_blend", poisson_blend_value.get()), save_switch_states())
    )
    poisson_blend_switch.grid(row=2, column=1, sticky="w", pady=4)
    ToolTip(poisson_blend_switch, "ผสมขอบรอบใบหน้าให้เนียนเรียบด้วยอัลกอริทึม Poisson")

    show_fps_value = ctk.BooleanVar(value=modules.globals.show_fps)
    show_fps_switch = ctk.CTkSwitch(
        sw_grid, text="แสดง FPS", variable=show_fps_value, cursor="hand2", font=ctk.CTkFont(size=12),
        command=lambda: (setattr(modules.globals, "show_fps", show_fps_value.get()), save_switch_states())
    )
    show_fps_switch.grid(row=3, column=0, sticky="w", pady=4)
    ToolTip(show_fps_switch, "แสดงตัวเลขเฟรมเรตต่อวินาทีบนมุมกล้องสด")

    color_correction_value = ctk.BooleanVar(value=modules.globals.color_correction)
    color_correction_switch = ctk.CTkSwitch(
        sw_grid, text="แก้สีกล้องอมฟ้า", variable=color_correction_value, cursor="hand2", font=ctk.CTkFont(size=12),
        command=lambda: (setattr(modules.globals, "color_correction", color_correction_value.get()), save_switch_states())
    )
    color_correction_switch.grid(row=3, column=1, sticky="w", pady=4)
    ToolTip(color_correction_switch, "ปรับแก้ทอนสีอมฟ้าอมเขียวจากกล้องเว็บแคมบางรุ่น")

    # ==================== LOWER SECTION: Webcam & Actions ====================
    bottom_frame = ctk.CTkFrame(root, corner_radius=10)
    bottom_frame.pack(fill="x", padx=15, pady=(5, 5))

    cam_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
    cam_frame.pack(fill="x", padx=15, pady=(10, 5))

    ctk.CTkLabel(cam_frame, text="เลือกรุ่นกล้องเว็บแคม:", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=(0, 10))

    available_cameras = get_available_cameras()
    camera_indices, camera_names = available_cameras

    if not camera_names or camera_names[0] == "No cameras found" or camera_names[0] == "ไม่พบกล้องเว็บแคม":
        camera_variable = ctk.StringVar(value="ไม่พบกล้องเว็บแคม")
        camera_optionmenu = ctk.CTkOptionMenu(cam_frame, variable=camera_variable, values=["ไม่พบกล้องเว็บแคม"], state="disabled", width=200)
    else:
        camera_variable = ctk.StringVar(value=camera_names[0])
        camera_optionmenu = ctk.CTkOptionMenu(cam_frame, variable=camera_variable, values=camera_names, width=200)

    camera_optionmenu.pack(side="left", fill="x", expand=True)
    ToolTip(camera_optionmenu, "เลือกรุ่นกล้องถ่ายภาพที่จะใช้งานในโหมดเรียลไทม์")

    live_button = ctk.CTkButton(
        cam_frame,
        text="📹 เปิดกล้องสด (Live)",
        font=ctk.CTkFont(size=13),
        cursor="hand2",
        width=140,
        fg_color="#059669",
        hover_color="#047857",
        command=lambda: webcam_preview(
            root,
            (
                camera_indices[camera_names.index(camera_variable.get())]
                if camera_names and camera_names[0] != "ไม่พบกล้องเว็บแคม" and camera_names[0] != "No cameras found"
                else None
            ),
        ),
        state=("normal" if camera_names and camera_names[0] != "ไม่พบกล้องเว็บแคม" and camera_names[0] != "No cameras found" else "disabled"),
    )
    live_button.pack(side="right", padx=(10, 0))
    ToolTip(live_button, "เริ่มสลับใบหน้าแบบเรียลไทม์สดผ่านกล้องเว็บแคม")

    action_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
    action_frame.pack(fill="x", padx=15, pady=(5, 10))

    start_button = ctk.CTkButton(
        action_frame,
        text="🚀 เริ่มสลับใบหน้า (START)",
        font=ctk.CTkFont(size=15, weight="bold"),
        cursor="hand2",
        height=40,
        fg_color="#2563EB",
        hover_color="#1D4ED8",
        command=lambda: analyze_target(start, root),
    )
    start_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
    ToolTip(start_button, "เริ่มกระบวนการสลับใบหน้าในวิดีโอหรือรูปภาพที่เลือกไว้")

    preview_button = ctk.CTkButton(
        action_frame,
        text="👁 ดูตัวอย่าง",
        font=ctk.CTkFont(size=13),
        cursor="hand2",
        height=40,
        width=110,
        fg_color="#4B5563",
        hover_color="#374151",
        command=lambda: toggle_preview(),
    )
    preview_button.pack(side="left", padx=5)
    ToolTip(preview_button, "แสดงหรือซ่อนหน้าต่างดูภาพตัวอย่าง")

    stop_button = ctk.CTkButton(
        action_frame,
        text="✖ ปิดโปรแกรม",
        font=ctk.CTkFont(size=13),
        cursor="hand2",
        height=40,
        width=100,
        fg_color="#DC2626",
        hover_color="#B91C1C",
        command=lambda: destroy(),
    )
    stop_button.pack(side="right", padx=(5, 0))
    ToolTip(stop_button, "หยุดการทำงานและปิดหน้าต่างโปรแกรม")

    global status_label
    status_bar = ctk.CTkFrame(root, height=30, fg_color="#111827", corner_radius=0)
    status_bar.pack(fill="x", side="bottom")

    status_label = ctk.CTkLabel(status_bar, text="พร้อมใช้งาน", font=ctk.CTkFont(size=12), text_color="#3B82F6", justify="center")
    status_label.pack(fill="both", expand=True, pady=3)

    return root


def close_mapper_window():
    global POPUP, POPUP_LIVE
    if POPUP and POPUP.winfo_exists():
        POPUP.destroy()
        POPUP = None
    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        POPUP_LIVE.destroy()
        POPUP_LIVE = None


def analyze_target(start: Callable[[], None], root: ctk.CTk):
    if POPUP != None and POPUP.winfo_exists():
        update_status("กรุณาจัดการหน้าต่าง Pop-up ให้เรียบร้อยก่อน")
        return

    if not modules.globals.source_path and not modules.globals.map_faces:
        update_status("กรุณาเลือกรูปภาพใบหน้าต้นฉบับก่อน!")
        return

    if not modules.globals.target_path:
        update_status("กรุณาเลือกวิดีโอหรือรูปภาพเป้าหมายก่อน!")
        return

    if modules.globals.map_faces:
        modules.globals.source_target_map = []

        if is_image(modules.globals.target_path):
            update_status("กำลังค้นหาใบหน้าในภาพเป้าหมาย...")
            get_unique_faces_from_target_image()
        elif is_video(modules.globals.target_path):
            update_status("กำลังค้นหาใบหน้าในวิดีโอเป้าหมาย...")
            get_unique_faces_from_target_video()

        if len(modules.globals.source_target_map) > 0:
            create_source_target_popup(start, root, modules.globals.source_target_map)
        else:
            update_status("ไม่พบใบหน้าในภาพ/วิดีโอเป้าหมาย")
    else:
        select_output_path(start)


def create_source_target_popup(
        start: Callable[[], None], root: ctk.CTk, map: list
) -> None:
    global POPUP, popup_status_label

    POPUP = ctk.CTkToplevel(root)
    POPUP.title(_("Source x Target Mapper"))
    POPUP.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")
    POPUP.focus()

    def on_submit_click(start):
        if has_valid_map():
            POPUP.destroy()
            select_output_path(start)
        else:
            update_pop_status("Atleast 1 source with target is required!")

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP, width=POPUP_SCROLL_WIDTH, height=POPUP_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_button_click(map, button_num):
        map = update_popup_source(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select source image"),
            command=lambda id=id: on_button_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=50, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        image = Image.fromarray(gpu_cvt_color(item["target"]["cv2"], cv2.COLOR_BGR2RGB))
        image = image.resize(
            (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        tk_image = ctk.CTkImage(image, size=image.size)

        target_image = ctk.CTkLabel(
            scrollable_frame,
            text=f"T-{id}",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        target_image.grid(row=id, column=3, padx=10, pady=10)
        target_image.configure(image=tk_image)

    popup_status_label = ctk.CTkLabel(POPUP, text=None, justify="center")
    popup_status_label.grid(row=1, column=0, pady=15)

    close_button = ctk.CTkButton(
        POPUP, text=_("Submit"), command=lambda: on_submit_click(start)
    )
    close_button.grid(row=2, column=0, pady=10)


def update_popup_source(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict

    source_path = ctk.filedialog.askopenfilename(
        title=_("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict[button_num].destroy()
        del source_label_dict[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = cv2.imread(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                gpu_cvt_color(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict[button_num] = source_image
        else:
            update_pop_status("Face could not be detected in last upload!")
        return map


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title(_("Preview"))
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=_(text))
    ROOT.update()


def update_pop_status(text: str) -> None:
    popup_status_label.configure(text=_(text))


def update_pop_live_status(text: str) -> None:
    popup_status_label_live.configure(text=_(text))


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, update the frame processors
    if PREVIEW.state() == "normal":
        global frame_processors
        frame_processors = get_frame_processors_modules(
            modules.globals.frame_processors
        )


def fetch_random_face() -> None:
    PREVIEW.withdraw()
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        }
        response = requests.get(
            "https://thispersondoesnotexist.com/",
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()

        if b"<html" in response.content[:200].lower():
            raise ValueError("The website returned HTML instead of an image (Cloudflare protection).")

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "deep_live_cam_random_face.jpg")
        with open(temp_path, "wb") as f:
            f.write(response.content)

        # Verify image validity
        with Image.open(temp_path) as img:
            img.verify()

        modules.globals.source_path = temp_path
        image = render_image_preview(temp_path, (200, 200))
        source_label.configure(image=image)
        update_status("Random face loaded!")
    except Exception as e:
        print(f"Failed to fetch random face: {e}")
        update_status("Failed to fetch random face. Please click 'Select a face' to choose a local image.")



def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(
        title=_("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)


def swap_faces_paths() -> None:
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET

    source_path = modules.globals.source_path
    target_path = modules.globals.target_path

    if not is_image(source_path) or not is_image(target_path):
        return

    modules.globals.source_path = target_path
    modules.globals.target_path = source_path

    RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
    RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)

    PREVIEW.withdraw()

    source_image = render_image_preview(modules.globals.source_path, (200, 200))
    source_label.configure(image=source_image)

    target_image = render_image_preview(modules.globals.target_path, (200, 200))
    target_label.configure(image=target_image)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(
        title=_("select an target image or video"),
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft, vid_ft],
    )
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if not modules.globals.source_path and not modules.globals.map_faces:
        update_status("กรุณาเลือกรูปภาพใบหน้าต้นฉบับก่อน!")
        return

    if not modules.globals.target_path:
        update_status("กรุณาเลือกวิดีโอหรือรูปภาพเป้าหมายก่อน!")
        return

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=_("save image output file"),
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=_("save video output file"),
            filetypes=[vid_ft],
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None

    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        
        def run_start_async():
            try:
                update_status("กำลังเริ่มกระบวนการสลับใบหน้า...")
                start()
            except Exception as e:
                update_status(f"เกิดข้อผิดพลาด: {e}")

        threading.Thread(target=run_start_async, daemon=True).start()


def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    """Check if the target is NSFW.
    TODO: Consider to make blur the target.
    """
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame

    if type(target) is str:  # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray:  # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(
                to_quit=False
            )  # Do not need to destroy the window frame if the target is NSFW
        update_status("Processing ignored!")
        return True
    else:
        return False


def fit_image_to_size(image, width: int, height: int):
    if width is None and height is None:
        return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    if width > height:
        ratio_h = height / h
    else:
        ratio_w = width / w
    ratio = max(ratio_w, ratio_h)
    new_size = (int(ratio * w), int(ratio * h))
    return gpu_resize(image, dsize=new_size)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
        video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == "normal":
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        for frame_processor in get_frame_processors_modules(
                modules.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
            )
        image = Image.fromarray(gpu_cvt_color(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        update_status("Processing succeed!")
        PREVIEW.deiconify()


def webcam_preview(root: ctk.CTk, camera_index: int):
    global POPUP_LIVE

    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        update_status("Source x Target Mapper is already open.")
        POPUP_LIVE.focus()
        return

    if not modules.globals.map_faces:
        if modules.globals.source_path is None:
            update_status("Please select a source image first")
            return
        create_webcam_preview(camera_index)
    else:
        modules.globals.source_target_map = []
        create_source_target_popup_for_webcam(
            root, modules.globals.source_target_map, camera_index
        )



def get_available_cameras():
    """Returns a list of available camera names and indices safely."""
    try:
        if platform.system() == "Windows":
            try:
                from pygrabber.dshow_graph import FilterGraph
                graph = FilterGraph()
                devices = graph.get_input_devices()
                if devices:
                    return list(range(len(devices))), devices
            except Exception as e:
                print(f"DirectShow camera detection skipped: {e}")
        return [0, 1], ["กล้อง 0", "กล้อง 1"]
    except Exception:
        return [0], ["กล้อง 0"]
    else:
        # Unix-like systems (Linux/Mac) camera detection
        camera_indices = []
        camera_names = []

        if platform.system() == "Darwin":
            # Do NOT probe cameras with cv2.VideoCapture on macOS — probing
            # invalid indices triggers the OBSENSOR backend and causes SIGSEGV.
            # Default to indices 0 and 1 (covers FaceTime + one USB camera).
            # The user can select the correct index from the UI dropdown.
            camera_indices = [0, 1]
            camera_names = ["Camera 0", "Camera 1"]
        else:
            # Linux camera detection - test first 10 indices
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()

        if not camera_names:
            return [], ["No cameras found"]

        return camera_indices, camera_names


def _capture_thread_func(cap, capture_queue, stop_event):
    """Capture thread: reads frames from camera and puts them into the queue.
    Drops frames when the queue is full to avoid backpressure on the camera."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        try:
            capture_queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame and enqueue the new one
            try:
                capture_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                capture_queue.put_nowait(frame)
            except queue.Full:
                pass


def _detection_thread_func(latest_frame_holder, detection_result, detection_lock, stop_event):
    """Detection thread: continuously runs face detection on the latest
    captured frame and stores results in detection_result under detection_lock.

    This decouples face detection (~15-30ms) from face swapping (~5-10ms)
    so the swap loop never blocks on detection, significantly improving
    live mode FPS."""
    while not stop_event.is_set():
        with detection_lock:
            frame = latest_frame_holder[0]

        if frame is None:
            time.sleep(0.005)
            continue

        if modules.globals.many_faces:
            many = get_many_faces(frame)
            with detection_lock:
                detection_result['target_face'] = None
                detection_result['many_faces'] = many
        else:
            face = get_one_face(frame)
            with detection_lock:
                detection_result['target_face'] = face
                detection_result['many_faces'] = None


def _processing_thread_func(capture_queue, processed_queue, stop_event,
                             latest_frame_holder, detection_result, detection_lock):
    """Processing thread: takes raw frames from capture_queue, reads the
    latest detection result from the shared detection_result dict, applies
    face swap/enhancement, and puts results into processed_queue.

    Face detection runs concurrently in _detection_thread_func — this thread
    only reads cached results so it never blocks on detection."""
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    source_image = None
    last_source_path = None
    prev_time = time.time()
    fps_update_interval = 0.5
    frame_count = 0
    fps = 0

    while not stop_event.is_set():
        try:
            frame = capture_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        temp_frame = frame

        if modules.globals.live_mirror:
            temp_frame = gpu_flip(temp_frame, 1)

        # Publish the mirrored frame for the detection thread to pick up
        with detection_lock:
            latest_frame_holder[0] = temp_frame

        if not modules.globals.map_faces:
            if modules.globals.source_path and modules.globals.source_path != last_source_path:
                last_source_path = modules.globals.source_path
                source_image = get_one_face(cv2.imread(modules.globals.source_path))

            # Read latest detection results (brief lock to avoid blocking detection thread)
            with detection_lock:
                cached_target_face = detection_result.get('target_face')
                cached_many_faces = detection_result.get('many_faces')

            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame(None, temp_frame)
                elif frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN256":
                    if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                        temp_frame = frame_processor.process_frame(None, temp_frame)
                elif frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN512":
                    if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                        temp_frame = frame_processor.process_frame(None, temp_frame)
                elif frame_processor.NAME == "DLC.FACE-SWAPPER":
                    # Use cached face positions from detection thread
                    swapped_bboxes = []
                    if modules.globals.many_faces and cached_many_faces:
                        result = temp_frame.copy()
                        for t_face in cached_many_faces:
                            result = frame_processor.swap_face(source_image, t_face, result)
                            if hasattr(t_face, 'bbox') and t_face.bbox is not None:
                                swapped_bboxes.append(t_face.bbox.astype(int))
                        temp_frame = result
                    elif cached_target_face is not None:
                        temp_frame = frame_processor.swap_face(source_image, cached_target_face, temp_frame)
                        if hasattr(cached_target_face, 'bbox') and cached_target_face.bbox is not None:
                            swapped_bboxes.append(cached_target_face.bbox.astype(int))
                    # Apply post-processing (sharpening, interpolation)
                    temp_frame = frame_processor.apply_post_processing(temp_frame, swapped_bboxes)
                else:
                    temp_frame = frame_processor.process_frame(source_image, temp_frame)
        else:
            modules.globals.target_path = None
            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                elif frame_processor.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                    fp_key = frame_processor.NAME.split(".")[-1].lower().replace("-", "_")
                    if modules.globals.fp_ui.get(fp_key, False):
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                else:
                    temp_frame = frame_processor.process_frame_v2(temp_frame)

        # Calculate and display FPS
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= fps_update_interval:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        if modules.globals.show_fps:
            cv2.putText(
                temp_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Put processed frame into output queue, dropping old frames if full
        try:
            processed_queue.put_nowait(temp_frame)
        except queue.Full:
            try:
                processed_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                processed_queue.put_nowait(temp_frame)
            except queue.Full:
                pass


def create_webcam_preview(camera_index: int):
    global preview_label, PREVIEW

    cap = VideoCapturer(camera_index)
    if not cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 60):
        update_status("Failed to start camera")
        return

    preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
    PREVIEW.deiconify()

    # Queues for decoupling capture from processing and processing from display.
    # Small maxsize ensures we always work on recent frames and drop stale ones.
    capture_queue = queue.Queue(maxsize=2)
    processed_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # Shared state for the detection pipeline.
    # latest_frame_holder[0] is the most recent raw frame for the detection
    # thread; detection_result holds the last detected faces for the
    # processing thread to read.  Both are guarded by detection_lock.
    detection_lock = threading.Lock()
    latest_frame_holder = [None]
    detection_result = {'target_face': None, 'many_faces': None}

    # Start capture thread
    cap_thread = threading.Thread(
        target=_capture_thread_func,
        args=(cap, capture_queue, stop_event),
        daemon=True,
    )
    cap_thread.start()

    # Start detection thread — runs face detection asynchronously so the
    # processing/swap thread never blocks on it
    det_thread = threading.Thread(
        target=_detection_thread_func,
        args=(latest_frame_holder, detection_result, detection_lock, stop_event),
        daemon=True,
    )
    det_thread.start()

    # Start processing thread
    proc_thread = threading.Thread(
        target=_processing_thread_func,
        args=(capture_queue, processed_queue, stop_event,
              latest_frame_holder, detection_result, detection_lock),
        daemon=True,
    )
    proc_thread.start()

    # Cleanup helper called from the display loop when preview closes
    def _cleanup():
        stop_event.set()
        cap_thread.join(timeout=2.0)
        det_thread.join(timeout=2.0)
        proc_thread.join(timeout=2.0)
        cap.release()
        PREVIEW.withdraw()

    # Non-blocking display loop using ROOT.after() — avoids blocking the
    # Tk event loop which could cause UI freezes or re-entrancy issues
    def _display_next_frame():
        if stop_event.is_set() or PREVIEW.state() == "withdrawn":
            _cleanup()
            return

        try:
            temp_frame = processed_queue.get_nowait()
        except queue.Empty:
            ROOT.after(16, _display_next_frame)
            return

        if modules.globals.live_resizable:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )
        else:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )

        image = gpu_cvt_color(temp_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageOps.contain(
            image, (temp_frame.shape[1], temp_frame.shape[0]), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)

        ROOT.after(16, _display_next_frame)

    # Kick off the non-blocking display loop
    ROOT.after(0, _display_next_frame)


def create_source_target_popup_for_webcam(
        root: ctk.CTk, map: list, camera_index: int
) -> None:
    global POPUP_LIVE, popup_status_label_live

    POPUP_LIVE = ctk.CTkToplevel(root)
    POPUP_LIVE.title(_("Source x Target Mapper"))
    POPUP_LIVE.geometry(f"{POPUP_LIVE_WIDTH}x{POPUP_LIVE_HEIGHT}")
    POPUP_LIVE.focus()

    def on_submit_click():
        if has_valid_map():
            simplify_maps()
            update_pop_live_status("Mappings successfully submitted!")
            create_webcam_preview(camera_index)  # Open the preview window
        else:
            update_pop_live_status("At least 1 source with target is required!")

    def on_add_click():
        add_blank_map()
        refresh_data(map)
        update_pop_live_status("Please provide mapping!")

    def on_clear_click():
        clear_source_target_images(map)
        refresh_data(map)
        update_pop_live_status("All mappings cleared!")

    popup_status_label_live = ctk.CTkLabel(POPUP_LIVE, text=None, justify="center")
    popup_status_label_live.grid(row=1, column=0, pady=15)

    add_button = ctk.CTkButton(POPUP_LIVE, text=_("Add"), command=lambda: on_add_click())
    add_button.place(relx=0.1, rely=0.92, relwidth=0.2, relheight=0.05)

    clear_button = ctk.CTkButton(POPUP_LIVE, text=_("Clear"), command=lambda: on_clear_click())
    clear_button.place(relx=0.4, rely=0.92, relwidth=0.2, relheight=0.05)

    close_button = ctk.CTkButton(
        POPUP_LIVE, text=_("Submit"), command=lambda: on_submit_click()
    )
    close_button.place(relx=0.7, rely=0.92, relwidth=0.2, relheight=0.05)



def clear_source_target_images(map: list):
    global source_label_dict_live, target_label_dict_live

    for item in map:
        if "source" in item:
            del item["source"]
        if "target" in item:
            del item["target"]

    for button_num in list(source_label_dict_live.keys()):
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    for button_num in list(target_label_dict_live.keys()):
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]


def refresh_data(map: list):
    global POPUP_LIVE

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP_LIVE, width=POPUP_LIVE_SCROLL_WIDTH, height=POPUP_LIVE_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_sbutton_click(map, button_num):
        map = update_webcam_source(scrollable_frame, map, button_num)

    def on_tbutton_click(map, button_num):
        map = update_webcam_target(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select source image"),
            command=lambda id=id: on_sbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=30, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select target image"),
            command=lambda id=id: on_tbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=3, padx=20, pady=10)

        if "source" in item:
            image = Image.fromarray(
                gpu_cvt_color(item["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=id, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)

        if "target" in item:
            image = Image.fromarray(
                gpu_cvt_color(item["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=id, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)


def update_webcam_source(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict_live

    source_path = ctk.filedialog.askopenfilename(
        title=_("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = cv2.imread(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                gpu_cvt_color(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict_live[button_num] = source_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map


def update_webcam_target(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global target_label_dict_live

    target_path = ctk.filedialog.askopenfilename(
        title=_("select an target image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "target" in map[button_num]:
        map[button_num].pop("target")
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]

    if target_path == "":
        return map
    else:
        cv2_img = cv2.imread(target_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["target"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                gpu_cvt_color(map[button_num]["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=button_num, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)
            target_label_dict_live[button_num] = target_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map