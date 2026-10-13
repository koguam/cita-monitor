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
CENTER = (540.0, 1065.0)

FONT_REGULAR = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
FONT_SEMIBOLD = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
FONT_BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"

# Same challenge for everyone: 24 rings, 5 seconds.
TIME_LIMIT = 5.0

_common = dict(
    rings=24,
    r0=136.0,          # radius of the innermost ring (centre line)
    pitch=12.0,        # distance between neighbouring rings
    gap_deg=62.0,      # angular size of the hole in every ring
    gravity=1400.0,    # px / s^2
)

ROUNDS = [
    dict(
        _common,
        key="chatgpt",
        title="Chat GPT",
        t_start=-1.69,      # the round "started" before the video begins
        t_show=0.0,         # first visible frame (the previous round ends here)
        t_up=3.31,          # timer hits zero -> freeze, YOU LOSE!
        outcome="lose",
        ring_width=9.0,
        ring_colors=["#F8DE7E", "#E9637E"],     # inner -> outer
        lose_color="#A30A0A",
        ball=dict(r=30.0, disc="#FFFFFF", logo="assets/logo_chatgpt_mask.png",
                  logo_color="#0D0D0D", logo_scale=1.0),
        restitution=(0.82, 1.0),
        # found by physics.search_rotation; delete to search again
        rotation=dict(theta_a=5.8262017, t_a=2.8247156, w0=-0.5818417, dw=0.0487310, d0=0.1244675),
        events=[
            (0.496, "bounce"),
            (1.248, "bounce"),
            (1.451, "escape"),
            (1.557, "escape"),
            (1.632, "escape"),
            (1.744, "escape"),
            (1.845, "escape"),
            (1.952, "escape"),
            (2.107, "bounce"),
            (2.944, "bounce"),
        ],
    ),
    dict(
        _common,
        key="claude",
        title="Claude",
        t_start=4.60,
        t_show=4.60,
        t_up=9.60,
        outcome="lose",
        ring_width=6.0,
        ring_colors=["#FFFFFF", "#B5242B"],
        lose_color="#B0070D",
        ball=dict(r=32.0, disc="#FFFFFF", logo="assets/logo_claude_mask.png",
                  logo_color="#D97757", logo_scale=1.0),
        restitution=(0.45, 0.75),
        rotation=dict(theta_a=4.8888350, t_a=7.6793026, w0=0.6148020, dw=0.0082976, d0=-0.1060473),
        events=[
            (5.401, "bounce"),
            (6.392, "bounce"),
            (6.979, "bounce"),
            (7.368, "settle"),   # lands softly and rolls -> stuck at the bottom
        ],
        # 9.02 - 9.60 the soundtrack ticks: the clock runs out
        tick_from=9.02,
    ),
    dict(
        _common,
        key="genora",
        title="Genora AI",
        t_start=11.00,
        t_show=11.00,
        t_up=16.00,          # only used if the ball would not make it
        outcome="win",
        ring_width=4.5,
        ring_colors=["#46E05A", "#2DB540"],
        glow=True,
        ball=dict(r=36.0, disc="genora", logo=None),
        restitution=(0.85, 1.0),
        bounce_penalty=0.6,
        rotation=dict(theta_a=2.9138807, t_a=15.8096783, w0=1.4550924, dw=0.0467621, d0=0.3419774),
        # every piano chord is a contact; the planner picks bounce or escape,
        # all 24 rings must be gone by the last chord
        events=[(t, "any") for t in (
            11.435, 11.808, 12.069, 12.485, 12.768, 13.285,
            13.504, 13.675, 14.517, 14.763, 15.152, 15.392)],
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
