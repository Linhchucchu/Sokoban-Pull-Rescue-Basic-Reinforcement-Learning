"""
High-Definition Game Graphics & Video Renderer for Sokoban Topic 15.
Renders polished 2D game frames with:
- 3D Beveled brick walls with authentic depth
- Polished stone floor parquet
- Classic reinforced wooden cargo crates (and gleaming golden crates on goal)
- Expressive worker character with directional gaze and safety gear
- Glowing mystical Hook recharge shrine (purple energy rune)
- Emerald beacon goal pads
- Dynamic rescue pull towing cables with energy beam effects
- Sleek HUD status dashboard (Pull credit, Hook status, Steps, Reward, Action badge)
- Victory banner overlays
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, Dict, Any


class GameRenderer:
    def __init__(self, cell_size: int = 72):
        self.cell_size = cell_size
        self.hud_height = 96
        self.margin = 16

        # Try to load a clean TTF font if available, otherwise default PIL font
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.font_tiny = None

        font_names = ["segoeui.ttf", "arial.ttf", "tahoma.ttf", "DejaVuSans.ttf"]
        for fn in font_names:
            try:
                self.font_large = ImageFont.truetype(fn, 24)
                self.font_medium = ImageFont.truetype(fn, 16)
                self.font_small = ImageFont.truetype(fn, 13)
                self.font_tiny = ImageFont.truetype(fn, 11)
                break
            except Exception:
                continue

        if self.font_large is None:
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.font_tiny = ImageFont.load_default()

    def render_frame(
        self,
        env,
        last_action: Optional[int] = None,
        last_event: str = "",
        step_num: int = 0,
        cum_reward: float = 0.0,
        is_solved: bool = False,
        is_terminated: bool = False,
        map_title: str = "Sokoban Topic 15",
    ) -> Image.Image:
        """
        Renders a full HD frame of the environment state.
        Returns a PIL Image (RGB).
        """
        grid_w = env.width * self.cell_size
        grid_h = env.height * self.cell_size

        total_width = grid_w + 2 * self.margin
        total_height = grid_h + self.hud_height + 2 * self.margin

        # Background: Modern deep navy/slate
        img = Image.new("RGB", (total_width, total_height), (15, 23, 42))  # #0F172A
        draw = ImageDraw.Draw(img)

        # 1. Render HUD Panel
        self._draw_hud(
            draw=draw,
            width=total_width,
            env=env,
            step_num=step_num,
            cum_reward=cum_reward,
            last_action=last_action,
            map_title=map_title,
        )

        # 2. Render Board (Floors, Walls, Goal, Hook)
        board_x = self.margin
        board_y = self.hud_height + self.margin

        # Draw board border shadow
        draw.rectangle(
            [board_x - 3, board_y - 3, board_x + grid_w + 2, board_y + grid_h + 2],
            outline=(51, 65, 85),  # #334155
            width=2,
        )

        for r in range(env.height):
            for c in range(env.width):
                x = board_x + c * self.cell_size
                y = board_y + r * self.cell_size

                if env.walls[r, c]:
                    self._draw_wall(draw, x, y)
                else:
                    self._draw_floor(draw, x, y, r, c)

                    # Draw Goal
                    if (r, c) == env.goal_pos:
                        self._draw_goal(draw, x, y)

                    # Draw Hook
                    if hasattr(env, "hook_pos") and env.hook_pos and (r, c) == env.hook_pos:
                        is_active = env.hook_used == 0
                        self._draw_hook(draw, x, y, is_active=is_active)

        # 3. Draw Towing Cable if last action was PULL
        if last_action == 4 or "pull" in last_event:
            self._draw_towing_cable(draw, board_x, board_y, env.agent_pos, env.box_pos)

        # 4. Draw Box (Crate)
        bx = board_x + env.box_pos[1] * self.cell_size
        by = board_y + env.box_pos[0] * self.cell_size
        on_goal = env.box_pos == env.goal_pos
        self._draw_box(draw, bx, by, on_goal=on_goal)

        # 5. Draw Agent
        ax = board_x + env.agent_pos[1] * self.cell_size
        ay = board_y + env.agent_pos[0] * self.cell_size
        self._draw_agent(draw, ax, ay, direction=last_action)

        # 6. Victory Overlay if solved
        if is_solved:
            self._draw_victory_overlay(draw, total_width, total_height, step_num, cum_reward)
        elif is_terminated and not is_solved:
            self._draw_deadlock_overlay(draw, total_width, total_height)

        return img

    def _draw_hud(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        env,
        step_num: int,
        cum_reward: float,
        last_action: Optional[int],
        map_title: str,
    ):
        """Draws a sleek, modern glassmorphism-style HUD at the top."""
        # Top panel background
        draw.rectangle([0, 0, width, self.hud_height], fill=(30, 41, 59))  # #1E293B
        draw.line([0, self.hud_height, width, self.hud_height], fill=(71, 85, 105), width=2)  # #475569

        # Title
        draw.text((16, 12), map_title, fill=(248, 250, 252), font=self.font_large)

        # Badges for Pull Credit and Hook
        if env.mode == "basic":
            self._draw_badge(draw, 16, 48, "MODE: BASIC (NO PULL)", (100, 116, 139), (255, 255, 255))
        else:
            # 1. Pull Credit Badge
            pull_avail = env.pull_available > 0
            p_badge_bg = (16, 185, 129) if pull_avail else (239, 68, 68)  # Emerald or Red
            p_text = f"PULL: READY ({env.pull_available})" if pull_avail else "PULL: USED (0)"
            self._draw_badge(draw, 16, 48, p_text, p_badge_bg, (255, 255, 255))

            # 2. Hook Tile Badge (Only if map actually has a Hook tile)
            if hasattr(env, "hook_pos") and env.hook_pos is not None:
                hook_avail = env.hook_used == 0
                h_badge_bg = (168, 85, 247) if hook_avail else (100, 116, 139)  # Purple or Gray
                h_text = "HOOK: READY (+1)" if hook_avail else "HOOK: CONSUMED"
                self._draw_badge(draw, 175, 48, h_text, h_badge_bg, (255, 255, 255))

        # Metrics on Right Side
        action_names = {0: "PUSH UP", 1: "PUSH DOWN", 2: "PUSH LEFT", 3: "PUSH RIGHT", 4: "RESCUE PULL"}
        act_str = action_names.get(last_action, "START")

        right_x = width - 180
        draw.text((right_x, 14), f"STEPS: {step_num}", fill=(203, 213, 225), font=self.font_medium)
        draw.text((right_x, 38), f"RETURN: {cum_reward:+.1f}", fill=(251, 191, 36), font=self.font_medium)
        draw.text((right_x, 62), f"ACT: {act_str}", fill=(147, 197, 253), font=self.font_small)

    def _draw_badge(self, draw, x, y, text, bg_color, text_color):
        pad_x, pad_y = 10, 4
        bbox = draw.textbbox((x + pad_x, y + pad_y), text, font=self.font_small)
        badge_w = bbox[2] - bbox[0] + 2 * pad_x
        badge_h = bbox[3] - bbox[1] + 2 * pad_y + 4
        draw.rounded_rectangle([x, y, x + badge_w, y + badge_h], radius=6, fill=bg_color)
        draw.text((x + pad_x, y + pad_y), text, fill=text_color, font=self.font_small)

    def _draw_wall(self, draw: ImageDraw.ImageDraw, x: int, y: int):
        """Draws a 3D beveled brick stone block."""
        cs = self.cell_size
        # Main body
        draw.rectangle([x, y, x + cs, y + cs], fill=(43, 45, 66))  # #2B2D42

        # 3D Bevel highlight on top and left
        draw.polygon([(x, y), (x + cs, y), (x + cs - 4, y + 4), (x + 4, y + 4), (x + 4, y + cs - 4), (x, y + cs)], fill=(71, 85, 105))

        # 3D Shadow on bottom and right
        draw.polygon([(x + cs, y), (x + cs, y + cs), (x, y + cs), (x + 4, y + cs - 4), (x + cs - 4, y + cs - 4), (x + cs - 4, y + 4)], fill=(15, 23, 42))

        # Inner brick mortar lines
        mid_y = y + cs // 2
        draw.line([x + 4, mid_y, x + cs - 4, mid_y], fill=(30, 41, 59), width=2)
        # Vertical mortar line row 1
        draw.line([x + cs // 2, y + 4, x + cs // 2, mid_y], fill=(30, 41, 59), width=2)
        # Vertical mortar line row 2
        draw.line([x + cs // 4, mid_y, x + cs // 4, y + cs - 4], fill=(30, 41, 59), width=2)
        draw.line([x + 3 * cs // 4, mid_y, x + 3 * cs // 4, y + cs - 4], fill=(30, 41, 59), width=2)

    def _draw_floor(self, draw: ImageDraw.ImageDraw, x: int, y: int, r: int, c: int):
        """Draws clean parquet stone floor tiles with subtle checker shading."""
        cs = self.cell_size
        # Subtle checkerboard variation
        base_shade = (241, 245, 249) if (r + c) % 2 == 0 else (248, 250, 252)
        draw.rectangle([x, y, x + cs, y + cs], fill=base_shade)
        # Subtle tile outline
        draw.rectangle([x, y, x + cs, y + cs], outline=(226, 232, 240), width=1)

    def _draw_goal(self, draw: ImageDraw.ImageDraw, x: int, y: int):
        """Draws an illuminated emerald beacon target pad."""
        cs = self.cell_size
        cx, cy = x + cs // 2, y + cs // 2
        r_outer = cs // 2 - 8
        r_inner = r_outer - 8

        # Glowing green outer circle
        draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], fill=(167, 243, 208), outline=(16, 185, 129), width=2)
        # Darker green inner circle
        draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=(16, 185, 129))
        # Center white ring
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 255))

    def _draw_hook(self, draw: ImageDraw.ImageDraw, x: int, y: int, is_active: bool):
        """Draws a mystic purple hook recharge rune shrine."""
        cs = self.cell_size
        cx, cy = x + cs // 2, y + cs // 2
        r = cs // 2 - 10

        if is_active:
            # Active glowing purple shrine
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(243, 232, 255), outline=(168, 85, 247), width=3)
            draw.ellipse([cx - r + 5, cy - r + 5, cx + r - 5, cy + r - 5], fill=(168, 85, 247))

            # Draw stylized anchor/hook icon
            draw.text((cx - 8, cy - 13), "H", fill=(255, 255, 255), font=self.font_medium)
            # Energy aura dots
            for angle in [0, 90, 180, 270]:
                rad = math.radians(angle)
                dot_x = cx + int((r + 4) * math.cos(rad))
                dot_y = cy + int((r + 4) * math.sin(rad))
                draw.ellipse([dot_x - 2, dot_y - 2, dot_x + 2, dot_y + 2], fill=(192, 132, 252))
        else:
            # Expended gray shrine
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(229, 231, 235), outline=(156, 163, 175), width=2)
            draw.text((cx - 6, cy - 10), "x", fill=(107, 114, 128), font=self.font_small)

    def _draw_box(self, draw: ImageDraw.ImageDraw, x: int, y: int, on_goal: bool):
        """Draws a reinforced classic cargo crate with metallic corner brackets."""
        cs = self.cell_size
        pad = 8
        bx, by = x + pad, y + pad
        bw, bh = cs - 2 * pad, cs - 2 * pad

        # Drop shadow
        draw.rounded_rectangle([bx + 3, by + 4, bx + bw + 3, by + bh + 4], radius=6, fill=(180, 190, 200))

        if on_goal:
            # Golden Crate
            fill_color = (251, 191, 36)      # Amber gold
            border_color = (217, 119, 6)     # Deep gold border
            cross_color = (245, 158, 11)     # Cross brace
            corner_color = (254, 243, 199)   # Shiny brass corner
        else:
            # Standard Wood Cargo Crate
            fill_color = (217, 119, 6)       # Rich wood
            border_color = (146, 64, 14)     # Dark oak border
            cross_color = (180, 83, 9)       # Wood brace
            corner_color = (100, 116, 139)   # Steel bracket

        # Main crate body
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=6, fill=fill_color, outline=border_color, width=3)

        # Cross braces (X shape)
        draw.line([bx + 4, by + 4, bx + bw - 4, by + bh - 4], fill=cross_color, width=4)
        draw.line([bx + bw - 4, by + 4, bx + 4, by + bh - 4], fill=cross_color, width=4)

        # 4 Metal Corner Reinforcements with Rivets
        corner_sz = 8
        # Top-Left
        draw.rectangle([bx, by, bx + corner_sz, by + corner_sz], fill=corner_color)
        draw.point((bx + 3, by + 3), fill=(255, 255, 255))
        # Top-Right
        draw.rectangle([bx + bw - corner_sz, by, bx + bw, by + corner_sz], fill=corner_color)
        draw.point((bx + bw - 4, by + 3), fill=(255, 255, 255))
        # Bottom-Left
        draw.rectangle([bx, by + bh - corner_sz, bx + corner_sz, by + bh], fill=corner_color)
        draw.point((bx + 3, by + bh - 4), fill=(255, 255, 255))
        # Bottom-Right
        draw.rectangle([bx + bw - corner_sz, by + bh - corner_sz, bx + bw, by + bh], fill=corner_color)
        draw.point((bx + bw - 4, by + bh - 4), fill=(255, 255, 255))

    def _draw_agent(self, draw: ImageDraw.ImageDraw, x: int, y: int, direction: Optional[int]):
        """Draws an expressive worker/keeper character with helmet, safety vest, and eyes."""
        cs = self.cell_size
        cx, cy = x + cs // 2, y + cs // 2

        # Soft drop shadow
        draw.ellipse([cx - 18, cy + 12, cx + 18, cy + 24], fill=(160, 175, 190))

        # Body (Torso with orange safety vest)
        draw.ellipse([cx - 18, cy - 6, cx + 18, cy + 20], fill=(37, 99, 235))  # Royal Blue jacket
        # Orange safety vest
        draw.rectangle([cx - 10, cy - 2, cx + 10, cy + 18], fill=(249, 115, 22))  # Orange vest
        # Reflective stripe
        draw.line([cx - 10, cy + 6, cx + 10, cy + 6], fill=(241, 245, 249), width=2)

        # Head
        head_radius = 14
        draw.ellipse([cx - head_radius, cy - 22, cx + head_radius, cy + 6], fill=(254, 215, 170))  # Skin tone

        # Hardhat / Helmet (Cyan/Blue)
        draw.ellipse([cx - head_radius - 2, cy - 26, cx + head_radius + 2, cy - 8], fill=(2, 132, 199))  # #0284C7
        # Helmet brim
        draw.rectangle([cx - head_radius - 3, cy - 12, cx + head_radius + 3, cy - 8], fill=(3, 105, 161))

        # Expressive Eyes looking towards movement direction
        eye_dx, eye_dy = 0, 0
        if direction == 0:  # UP
            eye_dy = -2
        elif direction == 1:  # DOWN
            eye_dy = 2
        elif direction == 2:  # LEFT
            eye_dx = -2
        elif direction == 3:  # RIGHT
            eye_dx = 2

        # Left Eye
        draw.ellipse([cx - 7 + eye_dx, cy - 7 + eye_dy, cx - 2 + eye_dx, cy - 2 + eye_dy], fill=(255, 255, 255))
        draw.point((cx - 4 + eye_dx, cy - 5 + eye_dy), fill=(15, 23, 42))

        # Right Eye
        draw.ellipse([cx + 2 + eye_dx, cy - 7 + eye_dy, cx + 7 + eye_dx, cy - 2 + eye_dy], fill=(255, 255, 255))
        draw.point((cx + 5 + eye_dx, cy - 5 + eye_dy), fill=(15, 23, 42))

    def _draw_towing_cable(self, draw: ImageDraw.ImageDraw, board_x: int, board_y: int, agent_pos, box_pos):
        """Draws an energized glowing towing cable during a RESCUE PULL."""
        cs = self.cell_size
        ax = board_x + agent_pos[1] * cs + cs // 2
        ay = board_y + agent_pos[0] * cs + cs // 2
        bx = board_x + box_pos[1] * cs + cs // 2
        by = board_y + box_pos[0] * cs + cs // 2

        # Outer glow
        draw.line([ax, ay, bx, by], fill=(6, 182, 212), width=6)  # Cyan glow
        # Core laser/cable
        draw.line([ax, ay, bx, by], fill=(255, 255, 255), width=2)  # White core

        # Sparkles along the cable
        mx, my = (ax + bx) // 2, (ay + by) // 2
        draw.ellipse([mx - 4, my - 4, mx + 4, my + 4], fill=(254, 240, 138))

    def _draw_victory_overlay(self, draw: ImageDraw.ImageDraw, width: int, height: int, steps: int, score: float):
        """Draws a celebratory victory banner overlay."""
        # Top banner
        bx, by = width // 2 - 160, height // 2 - 40
        draw.rounded_rectangle([bx, by, bx + 320, by + 80], radius=12, fill=(16, 185, 129), outline=(255, 255, 255), width=3)
        draw.text((bx + 35, by + 14), "★ PUZZLE SOLVED! ★", fill=(255, 255, 255), font=self.font_large)
        draw.text((bx + 40, by + 48), f"Optimal Path: {steps} Steps | Score: {score:+.1f}", fill=(240, 253, 244), font=self.font_small)

    def _draw_deadlock_overlay(self, draw: ImageDraw.ImageDraw, width: int, height: int):
        """Draws a deadlock warning overlay."""
        bx, by = width // 2 - 140, height // 2 - 35
        draw.rounded_rectangle([bx, by, bx + 280, by + 70], radius=12, fill=(239, 68, 68), outline=(255, 255, 255), width=2)
        draw.text((bx + 32, by + 12), "⚠ DEADLOCK TRAP ⚠", fill=(255, 255, 255), font=self.font_large)
        draw.text((bx + 40, by + 42), "Box Irreversibly Stuck!", fill=(254, 226, 226), font=self.font_small)
