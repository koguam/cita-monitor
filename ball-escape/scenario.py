"""Everything you might want to tweak: timings, colours, geometry.

All times are seconds on the soundtrack (assets/audio.m4a). Event times were
taken from the original video's audio with tools/onsets.py; every sound is
either a bounce of the ball, a ring the ball escapes through, or the landing
after which the ball rolls and gets stuck.
"""

W, H = 1080, 1920
FPS = 60
DURATION = 20.67
AUDIO = "assets/audio.m4a"
CENTER = (540.0, 1085.0)

# TikTok's own typeface (SIL OFL, assets/fonts/OFL.txt) for all on-screen text
FONT_TT_BOLD = "assets/fonts/TikTokSans-Bold.ttf"
FONT_TT_BLACK = "assets/fonts/TikTokSans-Black.ttf"
FONT_TT_SEMIBOLD = "assets/fonts/TikTokSans-SemiBold.ttf"

# Same challenge for everyone: 24 rings, 5 seconds.
TIME_LIMIT = 5.0

_common = dict(
    rings=24,
    pitch=11.0,        # distance between neighbouring rings
    gap_deg=62.0,      # angular size of the hole in every ring
    gravity=1400.0,    # px / s^2
)

# Free room for the ball's centre inside the first ring. Every bounce has to
# fit its flight time in here, so it stays the same however big the ball is:
# a bigger ball just gets a bigger first ring (r0).
SPACE = 100.0


def r0_for(ball_r, ring_width):
    return SPACE + ring_width / 2 + ball_r

ROUNDS = [
    dict(
        _common,
        key="chatgpt",
        # ring spin found by physics.search_rotation; delete to search again
        rotation=dict(theta_a=5.088498437154496, t_a=3.60864399956687, w0=-1.3608865174130718,
                      dw=-0.05330109059676037, d0=-0.10449568690031946),
        title="Chat GPT",
        t_start=-1.69,      # the round "started" before the video begins
        t_show=0.0,         # first visible frame (the previous round ends here)
        t_up=-1.69 + TIME_LIMIT,   # timer hits zero (3.31) -> freeze, YOU LOSE!
        outcome="lose",
        ring_width=9.0,
        r0=r0_for(70.0, 9.0),
        ring_colors=["#F8DE7E", "#E9637E"],     # inner -> outer
        lose_color="#A30A0A",
        ball=dict(r=70.0, disc="#FFFFFF", logo="assets/logo_chatgpt_mask.png",
                  logo_color="#0D0D0D", logo_scale=1.0),
        restitution=(0.82, 1.0),
        events=[
            (0.496, "bounce"),
            (1.248, "bounce"),
            # 1.45-2.04 is one continuous whistle: the ball dives into a hole
            # right on its start, then plain physics (nothing fitted) carries
            # it through the spiral, breaking ring after ring
            (1.451, "escape_free"),
            (2.107, "bounce"),
            (2.944, "bounce"),
        ],
        free_until=2.04,
        free_rings_min=12,
        free_spread_until=1.85,
    ),
    dict(
        _common,
        key="claude",
        # ring spin found by physics.search_rotation; delete to search again
        rotation=dict(theta_a=5.658104986200417, t_a=4.91372100192689, w0=0.40787764557968403,
                      dw=-0.018439207998415544, d0=0.10472819668253812),
        title="Claude",
        t_start=4.60,
        t_show=4.60,
        t_up=4.60 + TIME_LIMIT,
        outcome="lose",
        ring_width=6.0,
        r0=r0_for(72.0, 6.0),
        ring_colors=["#FFFFFF", "#B5242B"],
        lose_color="#B0070D",
        ball=dict(r=72.0, disc="#FFFFFF", logo="assets/logo_claude_mask.png",
                  logo_color="#D97757", logo_scale=1.0),
        restitution=(0.45, 0.75),
        events=[
            (5.401, "bounce"),
            (6.392, "bounce"),
            (6.979, "bounce"),
            (7.368, "settle"),   # lands softly and rolls at the bottom
        ],
        # 9.02 - 9.60 the soundtrack crackles: the spiral of holes reaches the
        # rolling ball, it drops through ring after ring (free physics, no
        # planning) and the clock runs out before it gets through them all
        drop_at=9.024,
        drop_rings=(12, 22),
    ),
    dict(
        _common,
        key="genora",
        # ring spin found by physics.search_rotation; delete to search again
        rotation=dict(theta_a=1.5313106355137427, t_a=14.84930669877302, w0=-0.5648491051519329,
                      dw=-0.044064360258183534, d0=0.08881007886043792),
        title="Genora AI",
        t_start=11.00,
        t_show=11.00,
        t_up=11.00 + TIME_LIMIT,   # only reached if the ball would not make it
        outcome="win",
        ring_width=4.5,
        r0=r0_for(80.0, 4.5),
        ring_colors=["#46E05A", "#2DB540"],
        glow=True,
        ball=dict(r=80.0, disc="genora", logo=None),
        restitution=(0.85, 1.0),
        bounce_penalty=0.6,
        # every piano chord is a contact; the planner picks bounce or escape,
        # all 24 rings must be gone by the last chord
        events=[(t, "any") for t in (
            11.435, 11.808, 12.069, 12.485, 12.768, 13.285,
            13.504, 13.675, 14.517, 14.763, 15.152, 15.392)],
        # climbing out of 24 rings on 12 fixed chords needs a bit more room
        # than the other rounds: up to x1.3 total speed per bounce and 400 px/s
        # of sliding slack (normal restitution still <= 1.3, no dead stops)
        limits=dict(E_CAP=1.3, SPEED_GAIN=1.3, SLIP_TOL=400.0),
    ),
]

END_CARD = dict(
    t_zoom=16.00,        # whoosh starts: zoom into the ball
    t_card=16.80,        # end card visible
    name="Genora AI",
    subtitle="Chatbot & AI Search",
    button="Install now",
    t_logo_draw=(16.90, 17.85),
    t_name=(17.85, 18.25),
    t_type=(18.30, 19.35),
    t_button=17.95,
)
