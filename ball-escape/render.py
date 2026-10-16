#!/usr/bin/env python3
"""Render the "ball escapes the rings" video.

    python3 render.py                 # full quality, output/ball_escape.mp4
    python3 render.py --preview       # half size, 30 fps, quick look
    python3 render.py --replan        # redo the physics search
    python3 render.py --still 2.0     # single frame as PNG

The physics plan is cached in plans.pkl per round. Changing a round in
scenario.py replans just that round; --replan searches every round again.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time

import cairo
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

HAVE_RAQM = features.check("raqm")      # OpenType features (tabular digits)

import physics as ph
import scenario as sc

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 2 * math.pi
_KEEP = []          # keep numpy buffers behind cached cairo surfaces alive
_KEEP_TMP = []      # per-frame buffers (glow), only the last few are needed


# --------------------------------------------------------------------------
# small helpers


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def mix(a, b, k):
    return tuple(a[i] + (b[i] - a[i]) * k for i in range(3))


def clamp01(x):
    return max(0.0, min(1.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3):
    x = clamp01(x)
    return 1 - (1 - x) ** p


def ease_in(x, p=3):
    return clamp01(x) ** p


def ease_out_back(x, s=1.70158):
    x = clamp01(x) - 1
    return x * x * ((s + 1) * x + s) + 1


def pil_to_surface(img):
    """RGBA PIL image -> premultiplied cairo ARGB32 surface."""
    a = np.asarray(img.convert("RGBA")).astype(np.float32)
    al = a[..., 3:4] / 255.0
    h, w = a.shape[:2]
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    buf = np.zeros((h, stride // 4, 4), np.uint8)
    buf[:, :w, 0] = (a[..., 2:3] * al)[..., 0]
    buf[:, :w, 1] = (a[..., 1:2] * al)[..., 0]
    buf[:, :w, 2] = (a[..., 0:1] * al)[..., 0]
    buf[:, :w, 3] = a[..., 3]
    _KEEP.append(buf)
    return cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_ARGB32, w, h, stride)


def surface_to_pil(surf):
    w, h = surf.get_width(), surf.get_height()
    buf = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
    return Image.fromarray(buf[..., [2, 1, 0, 3]].copy(), "RGBA")


class Fonts:
    cache = {}

    @classmethod
    def get(cls, path, size):
        key = (path, int(round(size)))
        if key not in cls.cache:
            full = path if os.path.isabs(path) else os.path.join(HERE, path)
            cls.cache[key] = ImageFont.truetype(full, key[1])
        return cls.cache[key]


class TextCache:
    """Pre-rendered text sprites. draw() centres the ink box on (x, y)."""
    cache = {}

    @classmethod
    def sprite(cls, text, path, size, color=(255, 255, 255), glow=0, glow_color=None,
               gradient=None):
        key = (text, path, size, color, glow, glow_color, gradient)
        if key in cls.cache:
            return cls.cache[key]
        f = Fonts.get(path, size)
        x0, y0, x1, y1 = f.getbbox(text)
        pad = int(glow * 3) + 6
        w, h = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).text((pad - x0, pad - y0), text, font=f, fill=255)
        if gradient:
            c0, c1 = np.array(gradient[0], float), np.array(gradient[1], float)
            k = np.linspace(0, 1, w)[None, :, None]
            col = (c0 + (c1 - c0) * k) * np.ones((h, 1, 1))
            img = Image.fromarray(np.dstack([col, np.asarray(mask)]).astype(np.uint8), "RGBA")
        else:
            img = Image.new("RGBA", (w, h), color + (0,))
            img.putalpha(mask)
        if glow:
            gc = glow_color or color
            g = Image.new("RGBA", (w, h), gc + (0,))
            g.putalpha(mask.filter(ImageFilter.GaussianBlur(glow)))
            base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            base.alpha_composite(g)
            base.alpha_composite(g)
            base.alpha_composite(img)
            img = base
        spr = (pil_to_surface(img), w, h, x1 - x0, y1 - y0)
        cls.cache[key] = spr
        return spr

    @classmethod
    def draw(cls, ctx, text, x, y, path, size, color=(255, 255, 255), alpha=1.0, scale=1.0,
             anchor="center", **kw):
        surf, w, h, iw, ih = cls.sprite(text, path, size, color, **kw)
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(scale, scale)
        ox = -w / 2 if anchor == "center" else (-(w - iw) / 2 if anchor == "left" else -w + (w - iw) / 2)
        ctx.set_source_surface(surf, ox, -h / 2)
        ctx.paint_with_alpha(alpha)
        ctx.restore()
        return iw, ih

    @classmethod
    def draw_tnum(cls, ctx, text, x, y, path, size, color=(255, 255, 255), alpha=1.0):
        """Tabular digits, centred on the advance width and the digit height."""
        key = ("tnum", text, path, size, color)
        if key not in cls.cache:
            f = Fonts.get(path, size)
            _x0, y0, _x1, y1 = f.getbbox("0123456789")
            pad = 8
            if HAVE_RAQM:
                adv = f.getlength(text, features=["tnum"])
                w, h = int(adv) + 2 * pad, (y1 - y0) + 2 * pad
                mask = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask).text((pad, pad - y0), text, font=f, fill=255, features=["tnum"])
            else:
                # no OpenType features available: centre each glyph in an
                # equal cell so the numbers still do not wobble
                cell = max(f.getlength(ch) for ch in "0123456789")
                cells = [cell if ch.isdigit() else f.getlength(ch) for ch in text]
                w, h = int(sum(cells)) + 2 * pad, (y1 - y0) + 2 * pad
                mask = Image.new("L", (w, h), 0)
                x = pad
                for ch, cw in zip(text, cells):
                    ImageDraw.Draw(mask).text((x + (cw - f.getlength(ch)) / 2, pad - y0), ch, font=f, fill=255)
                    x += cw
            img = Image.new("RGBA", (w, h), color + (0,))
            img.putalpha(mask)
            cls.cache[key] = (pil_to_surface(img), w, h)
        surf, w, h = cls.cache[key]
        ctx.set_source_surface(surf, x - w / 2, y - h / 2)
        ctx.paint_with_alpha(alpha)
        return w

    @classmethod
    def ink(cls, text, path, size):
        """Pixel-exact ink box (x0, x1) of a string drawn at the origin."""
        key = ("ink", text, path, size)
        if key not in cls.cache:
            f = Fonts.get(path, size)
            x0, _y0, x1, _y1 = f.getbbox(text)
            asc, desc = f.getmetrics()
            pad = 8
            mask = Image.new("L", (x1 - x0 + 2 * pad, asc + desc + 2 * pad), 0)
            ImageDraw.Draw(mask).text((pad - x0, pad), text, font=f, fill=255)
            bb = mask.getbbox() or (0, 0, 1, 1)
            cls.cache[key] = (bb[0] - pad + x0, bb[2] - pad + x0)
        return cls.cache[key]

    @classmethod
    def draw_cap(cls, ctx, text, x, y, path, size, color=(255, 255, 255), alpha=1.0):
        """Centre horizontally on the ink and vertically on the cap height."""
        key = ("cap", text, path, size, color)
        if key not in cls.cache:
            f = Fonts.get(path, size)
            ix0, ix1 = cls.ink(text, path, size)
            _a, cy0, _b, cy1 = f.getbbox("HXD")
            asc, desc = f.getmetrics()
            pad = 8
            w, h = int(ix1 - ix0) + 2 * pad, asc + desc + 2 * pad
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).text((pad - ix0, pad), text, font=f, fill=255)
            img = Image.new("RGBA", (w, h), color + (0,))
            img.putalpha(mask)
            cls.cache[key] = (pil_to_surface(img), w, pad + (cy0 + cy1) / 2)
        surf, w, yc = cls.cache[key]
        ctx.set_source_surface(surf, x - w / 2, y - yc)
        ctx.paint_with_alpha(alpha)

    @classmethod
    def width(cls, text, path, size):
        x0, _y0, x1, _y1 = Fonts.get(path, size).getbbox(text)
        return x1 - x0

    @classmethod
    def advance(cls, text, path, size):
        return Fonts.get(path, size).getlength(text)


def blur_glow(surf, scale=0.25, radius=6, boost=1.0):
    """Cheap bloom: downscale, blur, upscale. Returns a cairo surface."""
    w, h = surf.get_width(), surf.get_height()
    img = surface_to_pil(surf)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = img.resize((sw, sh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius))
    if boost != 1.0:
        a = np.asarray(small).astype(np.float32) * boost
        small = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
    big = small.resize((w, h), Image.BILINEAR)
    return pil_to_surface_premul(big)


def pil_to_surface_premul(img):
    """Image whose RGB is already premultiplied (came from a cairo surface)."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    buf = np.zeros((h, stride // 4, 4), np.uint8)
    buf[:, :w] = a[..., [2, 1, 0, 3]]
    _KEEP_TMP.append(buf)
    if len(_KEEP_TMP) > 8:
        del _KEEP_TMP[:4]
    return cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_ARGB32, w, h, stride)


# --------------------------------------------------------------------------
# physics plans (cached)


def plan_key(cfg):
    keep = {k: v for k, v in cfg.items() if k not in ("title", "ring_colors", "lose_color", "glow")}
    keep["center"] = sc.CENTER
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def get_plans(replan=False, samples=3000, seeds=(1, 4)):
    path = os.path.join(HERE, "plans.pkl")
    cache = {}
    if os.path.exists(path) and not replan:
        with open(path, "rb") as f:
            cache = pickle.load(f)
    plans = []
    changed = False
    for cfg in sc.ROUNDS:
        key = plan_key(cfg)
        if key not in cache:
            print(f"planning {cfg['key']} ...", flush=True)
            best = None
            if cfg.get("rotation") and not replan:
                best = ph.plan_round(cfg, sc.CENTER, ph.Rot(**cfg["rotation"]), beam=90,
                                     n_ang=120, n_esc=9, K=40, want_path=True)
            if best is None:
                for seed in seeds:
                    p = ph.search_rotation(cfg, sc.CENTER, n_samples=samples, seed=seed)
                    if p is not None and (best is None or p.cost < best.cost):
                        best = p
                if best is not None:
                    r = best.rot
                    print(f"  pin it in scenario.py: rotation=dict(theta_a={r.theta_a:.7f}, "
                          f"t_a={r.t_a:.7f}, w0={r.w0:.7f}, dw={r.dw:.7f}, d0={r.d0:.7f})")
            if best is None:
                sys.exit(f"no physical solution for round {cfg['key']}, loosen the scenario")
            t_end = cfg["t_up"] if cfg["outcome"] == "lose" else (best.t_free or cfg["t_up"])
            worst, where = ph.verify(best, cfg, sc.CENTER, cfg["t_show"], t_end)
            print(f"  {cfg['key']}: cost {best.cost:.3f}, max ring overlap {worst:.3f}px {where}")
            cache[key] = best
            changed = True
        plans.append(cache[key])
    live = {plan_key(cfg) for cfg in sc.ROUNDS}
    if changed or set(cache) != live:
        with open(path, "wb") as f:
            pickle.dump({k: v for k, v in cache.items() if k in live}, f)
    return plans


# --------------------------------------------------------------------------
# a round of the game


class RoundScene:
    def __init__(self, cfg, plan, seed=0):
        self.cfg = cfg
        self.plan = plan
        self.geo = ph.geo_from(cfg, sc.CENTER)
        self.rot = plan.rot
        n = self.geo.n
        c0, c1 = rgb(cfg["ring_colors"][0]), rgb(cfg["ring_colors"][1])
        self.ring_col = [mix(c0, c1, i / max(1, n - 1)) for i in range(n)]
        self.lose_col = rgb(cfg.get("lose_color", "#A00000"))
        self.broken = {}
        for (tb, i, _a, _s) in plan.breaks:
            self.broken[i] = tb
        self.t_free = plan.t_free
        self._build_debris(seed)
        self._build_spin()
        if cfg["ball"].get("logo"):
            m = Image.open(os.path.join(HERE, cfg["ball"]["logo"])).convert("L")
            img = Image.new("RGBA", m.size, (255, 255, 255, 0))
            img.putalpha(m)
            self.logo = pil_to_surface(img)
            self.logo_size = m.size[0]
        else:
            self.logo = None

    # ---- time helpers
    def frozen_time(self, t):
        """Physics time: stops at the time-up moment for a lost round."""
        if self.cfg["outcome"] == "lose":
            return min(t, self.cfg["t_up"])
        return t

    def remaining(self, t):
        cfg = self.cfg
        if self.cfg["outcome"] == "win" and self.t_free is not None and t >= self.t_free:
            return max(0.0, cfg["t_up"] - self.t_free)
        return max(0.0, cfg["t_up"] - t)

    # ---- debris: a broken ring turns into dots that fall away
    def _build_debris(self, seed):
        rng = np.random.default_rng(seed + 7)
        geo, rot = self.geo, self.rot
        self.debris = []
        for (tb, i, a_hit, sp) in self.plan.breaks:
            r = geo.r(i)
            th = rot.angle(i, tb)
            arc = TAU - geo.gap
            spacing = max(geo.lw * 1.1, 8.0)
            m = max(8, int(arc * r / spacing))
            ang = th + geo.gap / 2 + arc * (np.arange(m) + 0.5) / m
            nrm = np.stack([np.cos(ang), np.sin(ang)], -1)
            tan = np.stack([-np.sin(ang), np.cos(ang)], -1)
            p0 = geo.C + r * nrm
            w = rot.speed(i)
            burst = rng.uniform(25, 90, m)[:, None]
            v0 = w * r * tan + burst * nrm + rng.normal(0, 12, (m, 2))
            # a kick away from where the ball went through
            hit_dir = np.array([math.cos(a_hit), math.sin(a_hit)])
            near = np.clip(np.cos(ang - a_hit), 0, 1) ** 6
            v0 += (near * 220)[:, None] * hit_dir
            self.debris.append(dict(t=tb, ring=i, p0=p0, v0=v0, col=self.ring_col[i],
                                    size=max(2.4, geo.lw * 0.45),
                                    jit=rng.uniform(0.75, 1.15, m),
                                    life=float(rng.uniform(0.95, 1.2))))
        if self.plan.t_free is not None:
            # victory sparks from the ball as it leaves the last ring
            pos, vel = ph.ball_state(self.plan, geo, self.plan.t_free)
            for col, m, sp, size in (((0.55, 1.0, 0.7), 70, (250, 950), 3.6),
                                     ((1.0, 1.0, 1.0), 40, (150, 700), 2.6)):
                a = rng.uniform(0, TAU, m)
                s = rng.uniform(*sp, m)
                v0 = np.stack([np.cos(a), np.sin(a)], -1) * s[:, None] + 0.35 * vel
                self.debris.append(dict(t=self.plan.t_free, ring=-1, p0=np.repeat(pos[None], m, 0),
                                        v0=v0, col=col, size=size,
                                        jit=rng.uniform(0.6, 1.2, m), life=0.9))

    # ---- ball spin (rolling at every bounce)
    def _build_spin(self):
        """Ball spin: after every bounce it rolls along the wall, while rolling
        it turns with the rolling speed; through holes it keeps spinning."""
        geo, b, plan = self.geo, self.geo.b, self.plan
        pieces = sorted([(sg.t0, 0, sg) for sg in plan.segs] +
                        [(r["t"][0], 1, r) for r in plan.rolls], key=lambda x: x[0])
        first = pieces[0][0]
        om_prev = 0.0
        self.seg_om = {}
        for t0, kind, d in pieces:
            if kind == 0:
                if d.contact and t0 > first:
                    n = (d.p0 - geo.C) / max(1e-9, np.linalg.norm(d.p0 - geo.C))
                    om_prev = -(d.v0 @ np.array([-n[1], n[0]])) / b
                self.seg_om[id(d)] = om_prev
            else:
                om_prev = -d["R"] * d["om"][-1] / b
        end = (self.cfg["t_up"] if self.cfg["outcome"] == "lose" else (self.t_free or self.cfg["t_up"]) + 3) + 0.1
        dt = 1 / 600
        self.spin_t = np.arange(first, end, dt)
        oms = np.empty(len(self.spin_t))
        for i, t in enumerate(self.spin_t):
            piece = ph.active_piece(plan, t)
            if isinstance(piece, dict):
                j = min(int((t - piece["t"][0]) / ph.ROLL_DT), len(piece["t"]) - 1)
                oms[i] = -piece["R"] * piece["om"][j] / b
            else:
                oms[i] = self.seg_om[id(piece)]
        self.spin_a = np.concatenate([[0.0], np.cumsum(oms[:-1]) * dt])

    def spin(self, t):
        return float(np.interp(t, self.spin_t, self.spin_a))

    # ---- drawing
    def draw(self, ctx, t, ui=True):
        cfg = self.cfg
        tp = self.frozen_time(t)
        lose_k = 0.0
        if cfg["outcome"] == "lose" and t > cfg["t_up"]:
            lose_k = smooth((t - cfg["t_up"]) / 0.45)

        if cfg.get("glow"):
            layer = cairo.ImageSurface(cairo.FORMAT_ARGB32, sc.W, sc.H)
            lctx = cairo.Context(layer)
            self.draw_rings(lctx, tp, lose_k)
            self.draw_debris(lctx, tp, lose_k)
            glow = blur_glow(layer, scale=0.5, radius=2.5, boost=0.9)
            ctx.set_source_surface(glow, 0, 0)
            ctx.set_operator(cairo.OPERATOR_ADD)
            ctx.paint()
            ctx.set_operator(cairo.OPERATOR_OVER)
            ctx.set_source_surface(layer, 0, 0)
            ctx.paint()
        else:
            self.draw_rings(ctx, tp, lose_k)
            self.draw_debris(ctx, tp, lose_k)
        self.draw_ball(ctx, tp, lose_k)
        if ui:
            self.draw_ui(ctx, t, lose_k)

    def hit_flash(self, i, t):
        k = 0.0
        for (th, ri, _a, sp, _snd) in self.plan.hits:
            if ri == i and 0 <= t - th < 0.25:
                k = max(k, math.exp(-(t - th) / 0.07) * min(1.0, sp / 900))
        return k

    def draw_rings(self, ctx, t, lose_k):
        geo, rot = self.geo, self.rot
        cx, cy = geo.C
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        ctx.set_line_width(geo.lw)
        for i in range(geo.n):
            if i in self.broken and t >= self.broken[i]:
                continue
            th = rot.angle(i, t)
            col = self.ring_col[i]
            fl = self.hit_flash(i, t)
            if fl > 0:
                col = mix(col, (1, 1, 1), 0.55 * fl)
            if lose_k > 0:
                col = mix(col, self.lose_col, lose_k * (0.75 + 0.25 * i / geo.n))
            ctx.set_source_rgb(*col)
            ctx.new_path()
            ctx.arc(cx, cy, geo.r(i), th + geo.gap / 2, th + TAU - geo.gap / 2)
            ctx.stroke()

    def draw_debris(self, ctx, t, lose_k):
        g = np.array([0.0, self.geo.g])
        for d in self.debris:
            tau = t - d["t"]
            life = d["life"]
            if tau < 0 or tau > life:
                continue
            p = d["p0"] + d["v0"] * tau + 0.5 * g * tau * tau
            k = tau / life
            alpha = (1 - k) ** 1.3
            col = d["col"]
            if tau < 0.06:      # flash white for a moment so the break reads
                col = mix(col, (1, 1, 1), 0.65 * (1 - tau / 0.06))
            if lose_k > 0:
                col = mix(col, self.lose_col, lose_k)
            rad = d["size"] * (1 - 0.35 * k) * d["jit"]
            ctx.new_path()
            for j in range(len(p)):
                if -20 < p[j, 1] < sc.H + 20:
                    ctx.new_sub_path()
                    ctx.arc(p[j, 0], p[j, 1], rad[j], 0, TAU)
            ctx.set_source_rgba(col[0], col[1], col[2], alpha)
            ctx.fill()

    def draw_ball(self, ctx, t, lose_k):
        geo = self.geo
        pos, _v = ph.ball_state(self.plan, geo, t)
        if pos[1] > sc.H + 200:
            return
        b = geo.b
        spin = self.spin(t)
        draw_ball_sprite(ctx, self.cfg["ball"], pos[0], pos[1], b, spin, self.logo,
                         getattr(self, "logo_size", 512))
        if lose_k > 0:
            ctx.new_path()
            ctx.arc(pos[0], pos[1], b + 0.5, 0, TAU)
            ctx.set_source_rgba(*self.lose_col, 0.62 * lose_k)
            ctx.fill()

    # ---- title + countdown
    def draw_ui(self, ctx, t, _lose_k):
        cfg = self.cfg
        lost = cfg["outcome"] == "lose" and t >= cfg["t_up"]
        won = cfg["outcome"] == "win" and self.t_free is not None and t >= self.t_free
        if lost:
            k = (t - cfg["t_up"]) / 0.22
            pop = 1.0 + 0.3 * (1 - ease_out(k))
            tiktok_label(ctx, "YOU LOSE!", sc.W / 2, TITLE_Y, 80, fg=(255, 255, 255), bg=TT_RED,
                         scale=pop)
        else:
            tiktok_label(ctx, cfg["title"], sc.W / 2, TITLE_Y, 80)
        draw_timer(ctx, t, self.remaining(t), lost, won, cfg, self.t_free)
        if won:
            draw_escaped(ctx, t, self.t_free)


def draw_ball_sprite(ctx, ball, x, y, b, spin, logo=None, logo_size=512):
    if ball["disc"] == "genora":
        draw_genora_ball(ctx, x, y, b, spin)
        return
    ctx.new_path()
    ctx.arc(x, y, b, 0, TAU)
    ctx.set_source_rgb(*rgb(ball["disc"]))
    ctx.fill()
    if logo is not None:
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(spin)
        s = 2 * b * ball.get("logo_scale", 1.0) / logo_size
        ctx.scale(s, s)
        ctx.translate(-logo_size / 2, -logo_size / 2)
        ctx.set_source_rgb(*rgb(ball["logo_color"]))
        ctx.mask_surface(logo, 0, 0)
        ctx.restore()


def flower_path(ctx, rho, k_draw=1.0, stagger=True):
    """Six overlapping circles (the Genora mark). k_draw animates the stroke."""
    for j in range(6):
        a = j * TAU / 6 - math.pi / 2
        cx, cy = 0.82 * rho * math.cos(a), 0.82 * rho * math.sin(a)
        if stagger:
            kj = clamp01(k_draw * 1.6 - j * 0.12)
        else:
            kj = k_draw
        if kj <= 0:
            continue
        start = a + math.pi
        ctx.new_sub_path()
        ctx.arc(cx, cy, rho, start, start + TAU * kj)


def draw_genora_ball(ctx, x, y, b, spin):
    ctx.save()
    ctx.translate(x, y)
    # soft dark halo
    g = cairo.RadialGradient(0, 0, b * 0.9, 0, 0, b * 1.35)
    g.add_color_stop_rgba(0, 0.05, 0.16, 0.11, 0.85)
    g.add_color_stop_rgba(1, 0.05, 0.16, 0.11, 0.0)
    ctx.set_source(g)
    ctx.arc(0, 0, b * 1.35, 0, TAU)
    ctx.fill()
    g = cairo.RadialGradient(0, -b * 0.1, 0, 0, 0, b)
    g.add_color_stop_rgb(0, 0.42, 0.80, 0.58)
    g.add_color_stop_rgb(0.65, 0.20, 0.50, 0.35)
    g.add_color_stop_rgb(1, 0.10, 0.27, 0.19)
    ctx.set_source(g)
    ctx.arc(0, 0, b, 0, TAU)
    ctx.fill()
    ctx.rotate(spin)
    rho = b * 0.40
    for lw, a in ((b * 0.20, 0.10), (b * 0.12, 0.22), (b * 0.065, 0.95)):
        flower_path(ctx, rho, 1.0, stagger=False)
        ctx.set_line_width(lw)
        ctx.set_source_rgba(0.88, 1.0, 0.93, a)
        ctx.stroke()
    ctx.restore()


def rounded_rect(ctx, x, y, w, h, r):
    r = min(r, h / 2, w / 2)
    ctx.new_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    ctx.close_path()


TT_RED = (254, 44, 85)        # TikTok red
TT_GREEN = (59, 224, 122)
TITLE_Y = 300
TIMER_Y = 468


def tiktok_label(ctx, text, x, y, size, fg=(0, 0, 0), bg=(255, 255, 255), scale=1.0, alpha=1.0):
    """TikTok's in-app text style: bold text on a tight rounded box."""
    font = sc.FONT_TT_BOLD
    f = Fonts.get(font, size)
    x0, x1 = TextCache.ink(text, font, size)
    _a, cy0, _b, cy1 = f.getbbox("HXD")          # cap height, ignores descenders
    w = (x1 - x0) + 0.62 * size
    h = (cy1 - cy0) + 0.62 * size
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale, scale)
    rounded_rect(ctx, -w / 2, -h / 2, w, h, 0.26 * size)
    ctx.set_source_rgba(bg[0] / 255, bg[1] / 255, bg[2] / 255, alpha)
    ctx.fill()
    TextCache.draw_cap(ctx, text, 0, 0, font, size, color=fg, alpha=alpha)
    ctx.restore()


def draw_timer(ctx, t, remaining, lost, won, cfg, t_free):
    """Plain countdown: big digits, red in the last second."""
    if won:
        col = TT_GREEN
    elif lost or remaining < 1.0:
        col = TT_RED
    else:
        col = (255, 255, 255)
    pop = 1.0
    if lost and t - cfg["t_up"] < 0.25:
        pop = 1.0 + 0.25 * (1 - ease_out((t - cfg["t_up"]) / 0.25))
    if won and t - t_free < 0.3:
        pop = 1.0 + 0.25 * (1 - ease_out((t - t_free) / 0.3))
    ctx.save()
    ctx.translate(sc.W / 2, TIMER_Y)
    ctx.scale(pop, pop)
    TextCache.draw_tnum(ctx, f"{remaining:0.2f}", 0, 0, sc.FONT_TT_BLACK, 132, color=col)
    ctx.restore()


def draw_escaped(ctx, t, t_free):
    k = t - t_free
    if k < 0:
        return
    s = 0.5 + 0.5 * ease_out_back(k / 0.35, 2.2)
    tiktok_label(ctx, "ESCAPED!", sc.W / 2, sc.CENTER[1], 96, fg=(0, 0, 0), bg=TT_GREEN,
                 scale=s, alpha=ease_out(k / 0.15))


# --------------------------------------------------------------------------
# end card


class EndCard:
    def __init__(self):
        self.e = sc.END_CARD
        self.cx = sc.W / 2
        self.logo_y = 470
        self.rho = 92

    def draw_bg(self, ctx, alpha):
        ctx.save()
        g = cairo.RadialGradient(self.cx, self.logo_y + 120, 0, self.cx, self.logo_y + 120, 1000)
        g.add_color_stop_rgba(0, 0.07, 0.17, 0.12, alpha)
        g.add_color_stop_rgba(0.55, 0.02, 0.05, 0.035, alpha)
        g.add_color_stop_rgba(1, 0.0, 0.0, 0.0, alpha)
        ctx.set_source(g)
        ctx.paint()
        ctx.restore()

    def draw_logo(self, ctx, x, y, rho, k_draw, k_fill, glow_layer=True):
        # fill glow
        if k_fill > 0:
            g = cairo.RadialGradient(x, y, 0, x, y, rho * 1.9)
            g.add_color_stop_rgba(0, 0.36, 0.78, 0.52, 0.55 * k_fill)
            g.add_color_stop_rgba(0.6, 0.18, 0.48, 0.32, 0.35 * k_fill)
            g.add_color_stop_rgba(1, 0.05, 0.2, 0.12, 0.0)
            ctx.set_source(g)
            ctx.arc(x, y, rho * 1.9, 0, TAU)
            ctx.fill()
        ctx.save()
        ctx.translate(x, y)
        for lw, a in ((rho * 0.30, 0.07), (rho * 0.16, 0.16), (rho * 0.085, 0.35), (rho * 0.052, 0.95)):
            flower_path(ctx, rho, k_draw)
            ctx.set_line_width(lw)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            ctx.set_source_rgba(0.82, 1.0, 0.90, a)
            ctx.stroke()
        # centre star
        if k_fill > 0:
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            for j in range(6):
                a = j * TAU / 6
                ctx.move_to(0, 0)
                ctx.line_to(rho * 0.62 * math.cos(a), rho * 0.62 * math.sin(a))
            ctx.set_line_width(rho * 0.035)
            ctx.set_source_rgba(1, 1, 1, 0.85 * k_fill)
            ctx.stroke()
            g = cairo.RadialGradient(0, 0, 0, 0, 0, rho * 0.5)
            g.add_color_stop_rgba(0, 1, 1, 1, 0.9 * k_fill)
            g.add_color_stop_rgba(1, 1, 1, 1, 0)
            ctx.set_source(g)
            ctx.arc(0, 0, rho * 0.5, 0, TAU)
            ctx.fill()
        ctx.restore()

    def draw(self, ctx, t):
        e = self.e
        self.draw_bg(ctx, smooth((t - e["t_card"]) / 0.25))
        d0, d1 = e["t_logo_draw"]
        k_draw = ease_out((t - d0) / (d1 - d0), 2)
        k_fill = smooth((t - (d1 - 0.25)) / 0.4)
        # logo eases up a bit when the text arrives
        n0, n1 = e["t_name"]
        lift = ease_out((t - n0) / 0.5)
        rho = self.rho * (1.18 - 0.18 * lift)
        y = self.logo_y + 60 * (1 - lift)
        if t >= d0:
            self.draw_logo(ctx, self.cx, y, rho, k_draw, k_fill)
        # name
        k = ease_out((t - n0) / (n1 - n0))
        if k > 0:
            TextCache.draw(ctx, e["name"], self.cx, 795 + 25 * (1 - k), sc.FONT_TT_BOLD, 128,
                           alpha=k, gradient=((86, 190, 222), (52, 222, 128)))
        # subtitle typing + cursor
        s0, s1 = e["t_type"]
        if t >= s0 - 0.3:
            nchar = int(len(e["subtitle"]) * clamp01((t - s0) / (s1 - s0)))
            shown = e["subtitle"][:nchar]
            full_w = TextCache.width(e["subtitle"], sc.FONT_TT_SEMIBOLD, 62)
            x_left = self.cx - full_w / 2
            if shown:
                TextCache.draw(ctx, shown, x_left, 905, sc.FONT_TT_SEMIBOLD, 62, anchor="left")
            cw = TextCache.advance(shown, sc.FONT_TT_SEMIBOLD, 62) if shown else 0
            blink = (t < s1) or (int((t - s1) * 2.2) % 2 == 0)
            if blink:
                ctx.set_source_rgba(1, 1, 1, 0.95)
                ctx.rectangle(x_left + cw + 6, 905 - 34, 4, 68)
                ctx.fill()
        # button
        tb = e["t_button"]
        if t >= tb:
            k = (t - tb) / 0.45
            s = 1.0 + 0.45 * (1 - ease_out(k, 3))
            a = smooth(k * 2.5)
            if t > tb + 1.6:
                s *= 1 + 0.025 * math.sin((t - tb - 1.6) * 5.0)
            self.draw_button(ctx, self.cx, 1120, s, a)

    def draw_button(self, ctx, x, y, s, a):
        w, h = 620, 172
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(s, s)
        # glow
        for grow, al in ((26, 0.05), (16, 0.08), (8, 0.12)):
            rounded_rect(ctx, -w / 2 - grow, -h / 2 - grow, w + 2 * grow, h + 2 * grow, 34 + grow)
            ctx.set_source_rgba(0.15, 0.95, 0.6, al * a)
            ctx.fill()
        rounded_rect(ctx, -w / 2, -h / 2, w, h, 34)
        g = cairo.LinearGradient(0, -h / 2, 0, h / 2)
        g.add_color_stop_rgba(0, 0.17, 0.93, 0.63, a)
        g.add_color_stop_rgba(1, 0.06, 0.64, 0.42, a)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_line_width(3)
        ctx.set_source_rgba(0.55, 1.0, 0.8, 0.7 * a)
        ctx.stroke()
        TextCache.draw(ctx, self.e["button"], 0, 2, sc.FONT_TT_BOLD, 78, color=(245, 255, 250), alpha=a)
        ctx.restore()


# --------------------------------------------------------------------------
# timeline


class Video:
    def __init__(self, plans):
        self.rounds = [RoundScene(cfg, p, seed=i) for i, (cfg, p) in enumerate(zip(sc.ROUNDS, plans))]
        self.end = EndCard()

    def round_at(self, t):
        for i, r in enumerate(self.rounds):
            start = r.cfg["t_show"]
            stop = self.rounds[i + 1].cfg["t_show"] if i + 1 < len(self.rounds) else sc.END_CARD["t_card"]
            if start <= t < stop or (i == 0 and t < start):
                return r
        return None

    def draw(self, ctx, t):
        ctx.set_source_rgb(0, 0, 0)
        ctx.paint()
        e = sc.END_CARD
        if t < e["t_zoom"]:
            r = self.round_at(t)
            if r is not None:
                r.draw(ctx, t)
            return
        # whoosh: the arena rushes at the camera, a shock wave clears the screen
        last = self.rounds[-1]
        span = e["t_card"] - e["t_zoom"]
        k = (t - e["t_zoom"]) / span
        if k < 1.0:
            z = 1 + 0.8 * ease_in(k, 2)
            layer = cairo.ImageSurface(cairo.FORMAT_ARGB32, sc.W, sc.H)
            last.draw(cairo.Context(layer), t)
            ctx.save()
            ctx.translate(*sc.CENTER)
            ctx.scale(z, z)
            ctx.translate(-sc.CENTER[0], -sc.CENTER[1])
            ctx.set_source_surface(layer, 0, 0)
            ctx.paint_with_alpha(1 - smooth(k * 1.1))
            ctx.restore()
            cx, cy = sc.CENTER
            for dk, al in ((0.0, 0.8), (0.1, 0.4), (0.2, 0.2)):
                kk = (k - dk) / 0.9
                if 0 < kk < 1:
                    R = 180 + 1400 * ease_out(kk, 2)
                    for lw, aa in ((34, 0.15), (16, 0.35), (5, 1.0)):
                        ctx.new_path()
                        ctx.arc(cx, cy, R, 0, TAU)
                        ctx.set_line_width(lw * (1 - 0.6 * kk))
                        ctx.set_source_rgba(0.4, 1.0, 0.65, aa * al * (1 - kk) ** 1.3)
                        ctx.stroke()
            ctx.set_source_rgba(0.3, 0.95, 0.6, 0.10 * math.exp(-k * 6))
            ctx.paint()
        if t >= e["t_card"] - 0.05:
            self.end.draw(ctx, t)


def render(args):
    plans = get_plans(args.replan, samples=args.samples)
    video = Video(plans)
    scale = 0.5 if args.preview else 1.0
    fps = 30 if args.preview else sc.FPS
    W, H = int(sc.W * scale), int(sc.H * scale)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    ctx = cairo.Context(surf)
    ctx.scale(scale, scale)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)

    os.makedirs(args.outdir, exist_ok=True)
    if args.still is not None:
        for tstr in args.still.split(","):
            t = float(tstr)
            video.draw(ctx, t)
            surf.flush()
            out = os.path.join(args.outdir, f"still_{t:06.2f}.png")
            surf.write_to_png(out)
            print(out)
        return

    t0 = args.t_from
    t1 = args.t_to if args.t_to is not None else sc.DURATION
    os.makedirs(args.outdir, exist_ok=True)
    out = args.out or os.path.join(args.outdir, "ball_escape_preview.mp4" if args.preview else "ball_escape.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", os.path.join(HERE, sc.AUDIO),
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "veryfast" if args.preview else "slow",
           "-crf", "23" if args.preview else "16", "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "192k", "-shortest", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(round((t1 - t0) * fps))
    tic = time.time()
    for f in range(n):
        t = t0 + f / fps
        video.draw(ctx, t)
        surf.flush()
        proc.stdin.write(bytes(surf.get_data()))
        if f % fps == 0:
            print(f"\r  {t:5.2f}s  ({f}/{n}, {time.time() - tic:5.1f}s)", end="", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--replan", action="store_true")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--still", default=None, help="comma separated times -> PNGs")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--outdir", default=os.path.join(HERE, "output"))
    ap.add_argument("--out", default=None)
    render(ap.parse_args())


if __name__ == "__main__":
    main()
