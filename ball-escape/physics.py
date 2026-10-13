"""Ball-in-rotating-rings physics, planned so every contact lands on a sound.

Model
-----
* The ball is a disc of radius b under constant gravity g. Between contacts it
  flies on an exact parabola: p(t) = p0 + v0*t + g*t^2/2.
* Ring i is an annulus (centre radius r_i, width w) with a hole of angular size
  `gap`, rotating as theta_i(t) = theta_a + (w0 + i*dw) * (t - t_a).
  So the holes form a spiral that winds up / unwinds over time.
* The ball touches the inner side of ring i when |p - C| = r_i - w/2 - b.
  If at that moment the ball is lined up with the hole, it slips into it and
  the ring breaks (the trend's rule); otherwise it bounces.
* A bounce reflects the velocity: the normal part is multiplied by a
  restitution e in [e_min, e_max], the tangential part is kept (with a little
  friction allowed).

Sync to sound
-------------
Each event (time T, kind) must be a contact at exactly time T. From the
previous contact (p, t) the only parabola that reaches point q at time T is
v = (q - p - g*dt^2/2) / dt, so for every candidate landing angle we get one
outgoing velocity. A beam search picks the sequence of landing points whose
outgoing velocities are closest to a real reflection, rejecting any flight
that would touch a ring earlier, slip into a hole by accident, etc. On top of
that we search the ring rotation (phase, speed, twist) that makes the whole
round easiest, i.e. most physical.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

TAU = 2.0 * math.pi


def wrap(a):
    return (a + math.pi) % TAU - math.pi


@dataclass
class Rot:
    theta_a: float      # angle of ring 0's hole at time t_a
    t_a: float
    w0: float           # spin of ring 0 (rad/s)
    dw: float           # extra spin per ring -> the spiral winds/unwinds
    d0: float = 0.0     # hole offset between neighbouring rings at t_a

    def angle(self, i, t):
        return self.theta_a + i * self.d0 + (self.w0 + i * self.dw) * (t - self.t_a)

    def speed(self, i):
        return self.w0 + i * self.dw


@dataclass
class Geo:
    C: np.ndarray
    n: int
    r0: float
    pitch: float
    lw: float
    gap: float
    b: float
    g: float

    def r(self, i):
        return self.r0 + i * self.pitch

    def r_in(self, i):
        return self.r(i) - self.lw / 2

    def reff(self, i):
        return self.r_in(i) - self.b

    def esc_half(self, i, margin=0.03):
        """Angular half-width of the hole as seen by the ball's centre.

        The ring shatters the moment the ball touches it at the hole (its
        opening or the very edge of it), so the whole hole counts. Until that
        touch the ball is strictly inside, so nothing ever overlaps."""
        return self.gap / 2 - margin


def geo_from(cfg, center):
    return Geo(C=np.asarray(center, float), n=cfg["rings"], r0=cfg["r0"],
               pitch=cfg["pitch"], lw=cfg["ring_width"],
               gap=math.radians(cfg["gap_deg"]), b=cfg["ball"]["r"],
               g=cfg["gravity"])


# --------------------------------------------------------------------------
# result containers


@dataclass
class Seg:
    """Ballistic flight starting at t0 from p0 with velocity v0."""
    t0: float
    t1: float
    p0: np.ndarray
    v0: np.ndarray
    contact: bool = True      # starts with a bounce (False: passing a hole)


@dataclass
class Plan:
    cost: float
    rot: Rot
    segs: list = field(default_factory=list)
    breaks: list = field(default_factory=list)     # (t, ring index, angle)
    hits: list = field(default_factory=list)       # (t, ring index, angle, speed)
    roll: dict | None = None                       # rolling phase samples
    t_free: float | None = None                    # ball left the last ring
    event_costs: list = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers


def _bounce_cost(pos, vel, v, n_out, e_rng, k_min=0.8):
    """How far outgoing velocity v is from a physical reflection of vel.

    pos/vel: (S,2), v: (S,A,2), n_out: (S,2) outward normal at contact.
    """
    t_hat = np.stack([-n_out[:, 1], n_out[:, 0]], -1)
    vin_n = (vel * n_out).sum(-1)
    vin_t = (vel * t_hat).sum(-1)
    vo_n = (v * n_out[:, None]).sum(-1)
    vo_t = (v * t_hat[:, None]).sum(-1)
    lo_n = (-e_rng[1] * vin_n)[:, None]
    hi_n = (-e_rng[0] * vin_n)[:, None]
    dn = np.maximum(lo_n - vo_n, 0) + np.maximum(vo_n - hi_n, 0)
    a = (k_min * vin_t)[:, None]
    bb = vin_t[:, None]
    lo_t = np.minimum(a, bb)
    hi_t = np.maximum(a, bb)
    dt_ = np.maximum(lo_t - vo_t, 0) + np.maximum(vo_t - hi_t, 0)
    scale2 = (vin_n ** 2 + vin_t ** 2 + 250.0 ** 2)[:, None]
    cost = (dn ** 2 + dt_ ** 2) / scale2
    bad = vo_n > -15.0          # must actually leave the wall
    cost = np.where(bad, np.inf, cost)
    return cost


def _first_contact(p, v, gvec, C, R, t_max, steps=400):
    """Vectorised first time a parabola (starting inside) reaches radius R.

    p, v: (M,2). Returns tau (M,) (inf if none within t_max).
    """
    taus = np.linspace(0, t_max, steps + 1)[1:]
    x = p[:, None, :] + v[:, None, :] * taus[None, :, None] + 0.5 * gvec * taus[None, :, None] ** 2
    d = np.linalg.norm(x - C, axis=-1)
    hit = d >= R
    any_hit = hit.any(1)
    idx = np.argmax(hit, axis=1)
    lo = np.where(idx > 0, taus[np.maximum(idx - 1, 0)], 0.0)
    hi = taus[idx]
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        xm = p + v * mid[:, None] + 0.5 * gvec * mid[:, None] ** 2
        out = np.linalg.norm(xm - C, axis=-1) >= R
        hi = np.where(out, mid, hi)
        lo = np.where(out, lo, mid)
    return np.where(any_hit, hi, np.inf)


def roll_sim(geo, ring, t0, phi0, dphi0, t1, dt=1 / 600, damping=1.1):
    """Ball rolling on the inside of ring `ring` (no slipping, rolling drag)."""
    R = geo.reff(ring)
    k = (5.0 / 7.0) * geo.g / R
    n = int(math.ceil((t1 - t0) / dt)) + 1
    ts = t0 + np.arange(n) * dt
    phi = np.empty(n)
    om = np.empty(n)
    phi[0], om[0] = phi0, dphi0
    for j in range(1, n):
        # semi-implicit Euler is plenty at 600 Hz
        a = k * math.cos(phi[j - 1]) - damping * om[j - 1]
        om[j] = om[j - 1] + a * dt
        phi[j] = phi[j - 1] + om[j] * dt
    return ts, phi, om


# --------------------------------------------------------------------------
# planner


def _concat(batches):
    out = {}
    for key in batches[0]:
        vals = [b[key] for b in batches]
        if isinstance(vals[0], np.ndarray):
            out[key] = np.concatenate(vals, 0)
        else:
            out[key] = sum((list(v) for v in vals), [])
    return out


def plan_round(cfg, center, rot, beam=60, n_ang=72, n_esc=7, K=36,
               diversity=True, want_path=False, chain_tol=0.03, burst_win=0.075,
               bounce_win=0.10):
    """Beam search over landing points, one event (sound) at a time.

    An escape may break several rings in one flight when the holes line up:
    every ring crossed within `burst_win` of the sound belongs to that sound,
    a ring crossed within `chain_tol` of the *next* escape sound consumes that
    sound. After the last hole the ball must hit solid ring within
    `bounce_win` of the sound (so that bounce is heard as part of it) or land
    exactly on the next bounce sound. Anything else would be a silent event
    and is rejected.
    """
    geo = geo_from(cfg, center)
    C = geo.C
    gvec = np.array([0.0, geo.g])
    e_rng = cfg["restitution"]
    events = cfg["events"]
    E = len(events)
    t_up = cfg["t_up"]
    margin = 0.03
    win = cfg["outcome"] == "win"

    t0 = cfg["t_start"] if cfg["t_start"] >= cfg["t_show"] else cfg["t_show"]
    buckets = [[] for _ in range(E + 1)]
    buckets[0].append(dict(pos=np.array([C]), vel=np.zeros((1, 2)), t=np.array([t0]),
                           cost=np.zeros(1), init=np.array([True]), parent=np.array([-1]),
                           ring=np.array([0]), rland=np.array([0]), esc=np.array([False]),
                           k=np.array([-1]), t_from=np.array([t0]), p_from=np.array([C]),
                           v=np.zeros((1, 2)), T=np.array([t0]), q=np.array([C]),
                           v_land=np.zeros((1, 2)), brk=[[]]))
    nodes = []

    def keep_best(batch, n_keep):
        cost = batch["cost"]
        order = np.argsort(cost, kind="stable")
        order = order[np.isfinite(cost[order])]
        if diversity and len(order) > n_keep:
            pos = batch["pos"]
            bins = np.floor((np.arctan2(pos[:, 1] - C[1], pos[:, 0] - C[0]) % TAU)
                            / (TAU / 36)).astype(int) + 100 * batch["ring"]
            keep, per = [], {}
            for o in order:
                key = bins[o]
                if per.get(key, 0) < 2:
                    per[key] = per.get(key, 0) + 1
                    keep.append(o)
                    if len(keep) >= n_keep:
                        break
            order = np.array(keep, int)
        else:
            order = order[:n_keep]
        return order

    def terminal(pc, vc, tc, ring):
        """Extra cost for states that reached the end of the event list."""
        if win:
            return np.where(ring >= geo.n, 0.0, np.inf)
        out = np.full(len(pc), np.inf)
        for rr in np.unique(ring):
            sel = np.nonzero(ring == rr)[0]
            if rr >= geo.n:
                continue
            if events[-1][1] == "settle":
                out[sel] = _settle_cost(geo, rot, rr, pc[sel], vc[sel], tc[sel].max(), t_up, C)
            else:
                out[sel] = _coast_cost(geo, rot, rr, pc[sel], vc[sel], tc[sel], t_up, C, gvec, e_rng)
        return out

    for k in range(E + 1):
        if not buckets[k]:
            if k == E:
                return None
            continue
        batch = _concat(buckets[k])
        order = keep_best(batch, beam if k < E else 1)
        if len(order) == 0:
            if k == E:
                return None
            continue
        ids = []
        for o in order:
            nodes.append({key: val[o] for key, val in batch.items()})
            ids.append(len(nodes) - 1)
        if k == E:
            break
        ids = np.array(ids)
        T, kind = events[k]
        for rr in np.unique(batch["ring"][order]):
            selk = order[batch["ring"][order] == rr]
            sid = ids[batch["ring"][order] == rr]
            if rr >= geo.n:
                continue
            pos, vel, tt = batch["pos"][selk], batch["vel"][selk], batch["t"][selk]
            cost, init = batch["cost"][selk], batch["init"][selk]
            R = geo.reff(rr)
            th = rot.angle(rr, T)
            span = TAU - geo.gap - 2 * margin
            phis_b = th + geo.gap / 2 + margin + span * (np.arange(n_ang) + 0.5) / n_ang
            phis_e = th + np.linspace(-geo.esc_half(rr), geo.esc_half(rr), n_esc)
            if kind in ("bounce", "settle"):
                phis = phis_b
            elif kind == "escape":
                phis = phis_e
            else:   # "any": the planner decides, bounce or escape
                phis = np.concatenate([phis_b, phis_e])
            esc_a = np.zeros(len(phis), bool)
            if kind == "escape":
                esc_a[:] = True
            elif kind == "any":
                esc_a[len(phis_b):] = True
            A, S = len(phis), len(pos)
            u = np.stack([np.cos(phis), np.sin(phis)], -1)
            q = C + R * u
            dt = T - tt
            okdt = dt > 0.015
            dts = np.where(okdt, dt, 1.0)
            v = (q[None] - pos[:, None] - 0.5 * gvec * dts[:, None, None] ** 2) / dts[:, None, None]
            fr = np.concatenate([np.arange(1, K) / K, 1 - np.array([0.02, 0.008, 0.003])])
            taus = dts[:, None] * fr[None]
            x = (pos[:, None, None, :] + v[:, :, None, :] * taus[:, None, :, None]
                 + 0.5 * gvec * taus[:, None, :, None] ** 2)
            d = np.linalg.norm(x - C, axis=-1)
            ok = (d < R - 0.05).all(-1) & okdt[:, None]
            v_end = v + gvec * dts[:, None, None]
            ok &= ((q - C)[None] * v_end).sum(-1) > 1.0
            speed = np.linalg.norm(v, axis=-1)
            ok &= speed < 2600.0
            n_out = pos - C
            nn = np.linalg.norm(n_out, axis=-1, keepdims=True)
            n_out = n_out / np.where(nn > 1e-9, nn, 1.0)
            c_b = _bounce_cost(pos, vel, v, n_out, e_rng)
            c_init = 0.35 * (speed / 650.0) ** 2
            c = np.where(init[:, None], c_init, c_b) + 0.02 * (speed / 1400.0) ** 2
            total = cost[:, None] + np.where(ok, c, np.inf)
            flat = total.ravel()
            si = np.repeat(np.arange(S), A)
            ai = np.tile(np.arange(A), S)
            good = np.isfinite(flat)
            if not good.any():
                continue
            flat, si, ai = flat[good], si[good], ai[good]
            M = len(si)
            base = dict(parent=sid[si], t_from=tt[si], p_from=pos[si], v=v[si, ai],
                        T=np.full(M, T), q=q[ai], v_land=v_end[si, ai],
                        rland=np.full(M, rr), esc=esc_a[ai])

            def emit(jarr, mask, pc, vc, tc, ring, cc, brk, kk):
                for j in np.unique(jarr[mask]):
                    sel = np.nonzero(mask & (jarr == j))[0]
                    cj = cc[sel]
                    if j == E:
                        cj = cj + terminal(pc[sel], vc[sel], tc[sel], ring[sel])
                    fin = np.isfinite(cj)
                    sel, cj = sel[fin], cj[fin]
                    if len(sel) == 0:
                        continue
                    b = {key: val[sel] for key, val in base.items()}
                    b.update(pos=pc[sel], vel=vc[sel], t=tc[sel], cost=cj, ring=ring[sel],
                             init=np.zeros(len(sel), bool), brk=[brk[i] for i in sel],
                             k=kk[sel])
                    buckets[int(j)].append(b)

            pc, vc, tc = q[ai], v_end[si, ai], np.full(M, T)
            isb = ~esc_a[ai]
            if kind == "any":
                # chords that only bounce feel like no progress: nudge towards escapes
                flat = flat + np.where(isb, cfg.get("bounce_penalty", 0.0), 0.0)
            if isb.any():
                emit(np.full(M, k + 1), isb, pc, vc, tc, np.full(M, rr), flat,
                     [[] for _ in range(M)], np.full(M, k))
            if not (~isb).any():
                continue

            # ---- escape: follow the flight outwards through the holes
            brk = [[(T, int(rr), float(phis[a]))] for a in ai]
            consumed = np.full(M, k)          # last event index used
            t_evt = np.full(M, T)             # time of that event's sound
            alive = ~isb
            cur = rr
            while alive.any():
                nxt = cur + 1
                idx = np.nonzero(alive)[0]
                jn = consumed + 1
                if nxt >= geo.n:
                    # out of the last ring: the ball is free
                    okf = alive & (jn == E)
                    emit(jn, okf, pc, vc, tc, np.full(M, geo.n), flat, brk, consumed)
                    break
                R2 = geo.reff(nxt)
                tau = _first_contact(pc[idx], vc[idx], gvec, C, R2, 0.9)
                okc = np.isfinite(tau)
                tau = np.where(okc, tau, 0.0)
                p2 = pc[idx] + vc[idx] * tau[:, None] + 0.5 * gvec * tau[:, None] ** 2
                v2 = vc[idx] + gvec * tau[:, None]
                t2 = tc[idx] + tau
                ang = np.arctan2(p2[:, 1] - C[1], p2[:, 0] - C[0])
                rel = np.abs(wrap(ang - rot.angle(nxt, t2)))
                solid = np.zeros(M, bool)
                inwin = np.zeros(M, bool)
                solid[idx] = okc & (rel > geo.gap / 2 + margin)
                inwin[idx] = okc & (rel < geo.esc_half(nxt))
                P2, V2, T2 = pc.copy(), vc.copy(), tc.copy()
                P2[idx], V2[idx], T2[idx] = p2, v2, t2
                Tj = np.array([events[j][0] if j < E else np.inf for j in jn])
                kj = [events[j][1] if j < E else None for j in jn]
                is_b = np.array([x in ("bounce", "any") for x in kj])
                is_e = np.array([x in ("escape", "any") for x in kj])
                # (a) bounce heard as part of the escape sound
                ma = solid & (T2 - t_evt <= bounce_win) & ((jn == E) | (T2 < Tj - 0.02))
                late = 0.15 * ((T2 - t_evt) / bounce_win) ** 2   # tighter is better
                emit(jn, ma, P2, V2, T2, np.full(M, nxt), flat + late, brk, consumed)
                # (a2) the contact *is* the next bounce sound
                mb = solid & is_b & (np.abs(T2 - Tj) <= chain_tol) & ~ma
                emit(jn + 1, mb, P2, V2, T2, np.full(M, nxt), flat, brk, jn)
                # (b)/(c) slips through the next hole too
                mc = inwin & is_e & (np.abs(T2 - Tj) <= chain_tol)
                mbw = inwin & ~mc & (T2 - t_evt <= burst_win)
                # victory run: after the last sound the winner may fly straight
                # out through every hole that is lined up (while the chord rings)
                mvr = inwin & (jn == E) & win & (T2 < t_up - 0.05)
                go = mc | mbw | mvr
                if not go.any():
                    break
                for i in np.nonzero(go)[0]:
                    brk[i] = brk[i] + [(float(T2[i]), int(nxt), float(math.atan2(P2[i, 1] - C[1], P2[i, 0] - C[0])))]
                consumed = np.where(mc, jn, consumed)
                t_evt = np.where(mc, Tj, t_evt)
                pc, vc, tc = P2, V2, T2
                alive = go
                cur = nxt

    best = len(nodes) - 1
    out = Plan(cost=float(nodes[best]["cost"]), rot=rot)
    if not want_path:
        return out
    return _backtrack(cfg, geo, rot, nodes, best, out, gvec, C, e_rng)


def _settle_cost(geo, rot, ring, pos, vel, T, t_up, C, damping=1.1):
    """Landing that turns into rolling: the hole must never pass under it."""
    R = geo.reff(ring)
    half = geo.gap / 2 + math.asin(geo.b / geo.r_in(ring)) + 0.08
    n = (pos - C) / R
    t_hat = np.stack([-n[:, 1], n[:, 0]], -1)
    vin_n = (vel * n).sum(-1)
    vt = (vel * t_hat).sum(-1) * 5.0 / 7.0
    phi = np.arctan2(n[:, 1], n[:, 0])
    om = vt / R
    k = (5.0 / 7.0) * geo.g / R
    dt = 1 / 200
    bad = np.zeros(len(pos), bool)
    t = T
    while t < t_up:
        a = k * np.cos(phi) - damping * om
        om = om + a * dt
        phi = phi + om * dt
        t += dt
        bad |= np.abs(wrap(phi - rot.angle(ring, t))) < half
    # a soft landing: penalise slamming in fast, and wild swinging
    out = 0.15 * (np.maximum(vin_n - 450.0, 0) / 400.0) ** 2 + 0.05 * (np.abs(vt) / 400.0) ** 2
    return np.where(bad, np.inf, out)


def _coast_cost(geo, rot, ring, pos, vel, T, t_up, C, gvec, e_rng):
    """After the last bounce the ball must not touch anything before t_up."""
    out = np.full(len(pos), np.inf)
    R = geo.reff(ring)
    n = (pos - C) / np.linalg.norm(pos - C, axis=-1, keepdims=True)
    t_hat = np.stack([-n[:, 1], n[:, 0]], -1)
    vin_n = (vel * n).sum(-1)
    vin_t = (vel * t_hat).sum(-1)
    span = t_up - np.broadcast_to(np.asarray(T, float), (len(pos),))
    for e in np.linspace(e_rng[0], e_rng[1], 5):
        for kt in (1.0, 0.9, 0.8):
            vo = (-e * vin_n)[:, None] * n + (kt * vin_t)[:, None] * t_hat
            tau = _first_contact(pos, vo, gvec, C, R, float(span.max()) + 0.05, steps=200)
            ok = tau > span
            out = np.where(ok & np.isinf(out), 0.0, out)
    out = np.where(span <= 0.02, 0.0, out)
    return np.where(np.isinf(out), 50.0, out)


def _backtrack(cfg, geo, rot, nodes, best, out, gvec, C, e_rng):
    chain = []
    i = best
    while nodes[i]["parent"] >= 0:
        chain.append(nodes[i])
        i = int(nodes[i]["parent"])
    chain.reverse()
    segs, breaks, hits = [], [], []
    prev = 0.0
    out.event_costs = []
    for c in chain:
        out.event_costs.append(float(c["cost"]) - prev)
        prev = float(c["cost"])
        T = float(c["T"])
        segs.append(Seg(float(c["t_from"]), T, c["p_from"].copy(), c["v"].copy(),
                        contact=len(segs) > 0))
        ang = math.atan2(c["q"][1] - C[1], c["q"][0] - C[0])
        if c["esc"]:
            for (tb, rb, ab) in c["brk"]:
                breaks.append((tb, rb, ab, float(np.linalg.norm(c["v_land"]))))
            # one ballistic arc through the hole(s) until the next contact
            segs.append(Seg(T, float(c["t"]), c["q"].copy(), c["v_land"].copy(), contact=False))
            if int(c["ring"]) >= geo.n:
                out.t_free = float(c["brk"][-1][0])
            else:
                a2 = math.atan2(c["pos"][1] - C[1], c["pos"][0] - C[0])
                hits.append((float(c["t"]), int(c["ring"]), a2, float(np.linalg.norm(c["vel"])), True))
        else:
            hits.append((T, int(c["rland"]), ang, float(np.linalg.norm(c["v_land"])), True))
    last = chain[-1]
    ring_final = int(last["ring"])
    t_up = cfg["t_up"]
    p = last["pos"].copy()
    vin = last["vel"].copy()
    T = float(last["t"])
    if cfg["outcome"] == "lose":
        if cfg["events"][-1][1] == "settle":
            R = geo.reff(ring_final)
            n = (p - C) / R
            t_hat = np.array([-n[1], n[0]])
            vt = (vin @ t_hat) * 5.0 / 7.0
            phi0 = math.atan2(n[1], n[0])
            ts, phi, om = roll_sim(geo, ring_final, T, phi0, vt / R, t_up + 0.05)
            out.roll = dict(t=ts, phi=phi, om=om, R=R)
        else:
            # pick the most natural reflection that coasts to the freeze
            n = (p - C) / np.linalg.norm(p - C)
            t_hat = np.array([-n[1], n[0]])
            vin_n, vin_t = vin @ n, vin @ t_hat
            R = geo.reff(ring_final)
            chosen = None
            emid = 0.5 * (e_rng[0] + e_rng[1])
            cands = sorted([(abs(e - emid) + abs(1 - kt), e, kt)
                            for e in np.linspace(e_rng[0], e_rng[1], 5) for kt in (1.0, 0.9, 0.8)])
            for _, e, kt in cands:
                vo = -e * vin_n * n + kt * vin_t * t_hat
                tau = _first_contact(p[None], vo[None], gvec, C, R, t_up - T + 0.05, steps=300)[0]
                if tau > t_up - T:
                    chosen = vo
                    break
            if chosen is None:
                chosen = -emid * vin_n * n + vin_t * t_hat
            segs.append(Seg(T, t_up + 1.0, p, chosen))
    else:
        # free flight after the last ring is gone
        segs[-1] = Seg(segs[-1].t0, 99.0, segs[-1].p0, segs[-1].v0, contact=False)
    out.segs = segs
    out.breaks = breaks
    out.hits = hits
    return out


# --------------------------------------------------------------------------
# trajectory evaluation + verification


def ball_state(plan: Plan, geo: Geo, t: float):
    """Position and velocity of the ball at time t."""
    if plan.roll is not None and t >= plan.roll["t"][0]:
        r = plan.roll
        j = min(int((t - r["t"][0]) / (r["t"][1] - r["t"][0])), len(r["t"]) - 1)
        phi, om = r["phi"][j], r["om"][j]
        n = np.array([math.cos(phi), math.sin(phi)])
        return geo.C + r["R"] * n, r["R"] * om * np.array([-n[1], n[0]])
    gvec = np.array([0.0, geo.g])
    seg = plan.segs[0]
    for s in plan.segs:
        if t >= s.t0:
            seg = s
        else:
            break
    tau = max(t - seg.t0, 0.0)
    return seg.p0 + seg.v0 * tau + 0.5 * gvec * tau * tau, seg.v0 + gvec * tau


def verify(plan: Plan, cfg, center, t0, t1, dt=0.0005):
    """Return the worst overlap (px) between the ball and any intact ring."""
    geo = geo_from(cfg, center)
    broken = {}
    for (tb, i, _a, _s) in plan.breaks:
        broken[i] = tb
    worst = 0.0
    where = None
    for t in np.arange(t0, t1, dt):
        p, _ = ball_state(plan, geo, t)
        d = np.linalg.norm(p - geo.C)
        ang = math.atan2(p[1] - geo.C[1], p[0] - geo.C[0])
        for i in range(geo.n):
            if i in broken and t >= broken[i] - 1e-6:
                continue
            r = geo.r(i)
            if abs(d - r) > geo.lw / 2 + geo.b + 1:
                continue
            th = plan.rot.angle(i, t)
            rel = wrap(ang - th)
            if abs(rel) > geo.gap / 2:
                pen = geo.lw / 2 + geo.b - abs(d - r)
            else:
                # distance to the two end caps of the hole
                pen = -1e9
                for s in (-1, 1):
                    a = th + s * geo.gap / 2
                    u = np.array([math.cos(a), math.sin(a)])
                    proj = np.clip((p - geo.C) @ u, geo.r_in(i), geo.r_in(i) + geo.lw)
                    dist = np.linalg.norm(p - (geo.C + proj * u))
                    pen = max(pen, geo.b - dist)
            if pen > worst:
                worst, where = pen, (round(float(t), 4), i)
    return worst, where


# --------------------------------------------------------------------------
# search over ring rotations


def _sample_rot(cfg, rng):
    t_lo, t_hi = cfg["t_show"] - 1.0, cfg["t_up"] + 1.0
    while True:
        w0 = rng.choice([-1, 1]) * rng.uniform(*cfg.get("spin_range", (0.4, 1.5)))
        d0 = rng.choice([-1, 1]) * rng.uniform(*cfg.get("offset_range", (0.08, 0.36)))
        dw = rng.uniform(-0.06, 0.06)
        rot = Rot(theta_a=rng.uniform(0, TAU), t_a=rng.uniform(t_lo, t_hi), w0=w0, dw=dw, d0=d0)
        # looks: the holes should form one clean spiral arm the whole round,
        # i.e. the offset between neighbouring rings stays small
        lim = cfg.get("max_offset", 0.36)
        eff = [abs(d0 + dw * (tt - rot.t_a)) for tt in (cfg["t_show"], cfg["t_up"])]
        if max(eff) <= lim:
            return rot


def _coarse(args):
    cfg, center, seed, n = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rot = _sample_rot(cfg, rng)
        p = plan_round(cfg, center, rot, beam=24, n_ang=48, n_esc=7, K=24, diversity=False)
        if p is not None and math.isfinite(p.cost):
            out.append((p.cost, rot))
    return out


def _fine(args):
    cfg, center, rot = args
    return plan_round(cfg, center, rot, beam=90, n_ang=120, n_esc=9, K=40, want_path=True)


def search_rotation(cfg, center, n_samples=300, seed=1, keep=8, log=print, workers=None):
    """Random search over ring rotations, then a finer beam search on the best."""
    from concurrent.futures import ProcessPoolExecutor
    workers = workers or os.cpu_count() or 1
    chunks = [(cfg, center, seed * 1000 + w, n_samples // workers + 1) for w in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        cands = [c for part in ex.map(_coarse, chunks) for c in part]
        cands.sort(key=lambda c: c[0])
        log(f"  {cfg['key']}: {len(cands)}/{n_samples} feasible"
            + (f", best coarse cost {cands[0][0]:.3f}" if cands else ""))
        fines = list(ex.map(_fine, [(cfg, center, rot) for _c, rot in cands[:keep]]))
    best = None
    for (c0, rot), p in zip(cands[:keep], fines):
        if p is None:
            continue
        log(f"    refine {c0:.3f} -> {p.cost:.3f}")
        if best is None or p.cost < best.cost:
            best = p
    return best
