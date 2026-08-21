#!/usr/bin/env python3
"""
Cita Previa Extranjería monitor -> Telegram.

Watches the Spanish Sede Electrónica for free appointment slots for a single
trámite in a single province and pushes a Telegram message when slots appear.

It only ever READS availability. It never books, never picks a slot and never
submits contact details -- booking stays a manual step for the human.

The site sits behind an F5 WAF that starts rejecting requests if you poke it
too often, so the loop detects a block and backs off instead of hammering.
"""
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests
from playwright.sync_api import sync_playwright

BASE = "https://icp.administracionelectronica.gob.es"

# --- config (env) ---------------------------------------------------------
PROVINCE = os.environ.get("PROVINCE", "43")            # 43 = Tarragona
CATEGORY = os.environ.get("CATEGORY", "icpplus")       # icpplus for Tarragona
TRAMITE_PARAM = os.environ.get("TRAMITE_PARAM", "tramiteGrupo[1]")
TRAMITE = os.environ.get("TRAMITE", "4112")            # POLICÍA TARJETA CONFLICTO UCRANIA
TRAMITE_NAME = os.environ.get("TRAMITE_NAME", "POLICÍA TARJETA CONFLICTO UCRANIA")
SEDE = os.environ.get("SEDE", "99")                    # 99 = cualquier oficina
PROVINCE_NAME = os.environ.get("PROVINCE_NAME", "Tarragona")

NIE = os.environ.get("NIE", "")
FULL_NAME = os.environ.get("FULL_NAME", "")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_IDS = [c.strip() for c in os.environ.get("CHAT_IDS", "").split(",") if c.strip()]

# Optional Spanish proxy, as an alternative to the VPN hop.
PROXY_SERVER = os.environ.get("PROXY_SERVER", "")
PROXY_USER = os.environ.get("PROXY_USER", "")
PROXY_PASS = os.environ.get("PROXY_PASS", "")

INTERVAL = int(os.environ.get("INTERVAL_SECONDS", "300"))
REMIND_AFTER = int(os.environ.get("REMIND_AFTER_SECONDS", "1800"))
LOOP_MINUTES = int(os.environ.get("LOOP_MINUTES", "0"))
# Shell command that reconnects the VPN through another Spanish node.
ROTATE_CMD = os.environ.get("ROTATE_CMD", "")
MAX_ROTATIONS = int(os.environ.get("MAX_ROTATIONS", "4"))

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# Result codes
AVAILABLE, NO_SLOTS, BLOCKED, ERROR = "AVAILABLE", "NO_SLOTS", "BLOCKED", "ERROR"

BLOCK_MARKERS = (
    "The requested URL was rejected",
    "Request Rejected",
    "support ID is",
    "FortiGate",
    "Intrusion Prevention",
)


class Blocked(Exception):
    """The WAF turned us away."""


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sleep_ms(lo, hi):
    time.sleep(random.uniform(lo, hi) / 1000.0)


def send_telegram(text):
    if not TELEGRAM_TOKEN or not CHAT_IDS:
        log("Telegram not configured, skipping notification")
        return
    for chat_id in CHAT_IDS:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": text,
                      "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=30,
            )
            if r.status_code == 200:
                log(f"Telegram -> {chat_id} ok")
            else:
                log(f"Telegram -> {chat_id} FAILED {r.status_code}: {r.text[:200]}")
        except Exception as e:
            log(f"Telegram -> {chat_id} error: {e}")


def new_browser(p):
    """Real Chrome. Playwright's bundled headless shell is rejected by the
    FortiGate IPS in front of the site; the real Chrome build passes."""
    args = [
        "--disable-blink-features=AutomationControlled",
        "--ignore-certificate-errors",
        "--disable-gpu",
        "--no-sandbox",
    ]
    kw = {"headless": True, "args": args}
    if PROXY_SERVER:
        kw["proxy"] = {"server": PROXY_SERVER}
        if PROXY_USER:
            kw["proxy"]["username"] = PROXY_USER
            kw["proxy"]["password"] = PROXY_PASS
    try:
        return p.chromium.launch(channel="chrome", **kw)
    except Exception as e:
        log(f"channel=chrome unavailable ({e}); falling back to bundled chromium")
        return p.chromium.launch(**kw)


def body_of(pg):
    try:
        return pg.inner_text("body")
    except Exception:
        return ""


def guard(pg, where):
    """Raise if the WAF served a rejection instead of the real page."""
    text = body_of(pg)
    for marker in BLOCK_MARKERS:
        if marker in text:
            raise Blocked(f"{where}: {marker}")
    return text


def wait_for(pg, selector, where, timeout=30000):
    """wait_for_selector, but report a WAF block as a block rather than a
    mystery timeout."""
    try:
        pg.wait_for_selector(selector, timeout=timeout)
    except Exception:
        guard(pg, where)                     # raises Blocked if that's the cause
        raise RuntimeError(f"{where}: {selector} never appeared "
                           f"(page starts: {body_of(pg)[:160]!r})")


NO_SLOT_MARKERS = (
    "no hay citas disponibles",
    "En este momento no hay citas",
)


def classify(pg, text):
    """Decide what the post-solicitud page is saying, or None if not there yet.

    Structure beats wording: the office dropdown (#idSede) and the slot radios
    only exist when there is something bookable, whereas the exact Spanish
    sentence around them varies between trámites and site versions.
    """
    low = text.lower()
    if any(m.lower() in low for m in NO_SLOT_MARKERS):
        return NO_SLOTS

    try:
        if pg.query_selector("#idSede") or pg.query_selector(
                "input[type=radio][name=rdbCita]"):
            return AVAILABLE
    except Exception:
        pass

    if ("seleccione la oficina" in low
            or "citas disponibles" in low
            or "dispone de 5 minutos" in low):
        return AVAILABLE
    return None


def offices_on_offer(pg):
    """Names of the offices the site is offering, read off the page we already
    have. Tells the human where the slot is without costing another request."""
    try:
        names = pg.eval_on_selector_all(
            "#idSede option", "els => els.map(e => e.textContent)")
    except Exception:
        return []
    out = []
    for n in names:
        n = (n or "").strip()
        if n and "eleccionar" not in n and "ualquier" not in n:
            out.append(n)
    return out


def settle(pg, timeout=25000):
    """Wait for subresources, not just the HTML.

    The page's behaviour lives in a JS bundle. Over the VPN the runner is slow
    enough that domcontentloaded fires long before that bundle arrives, and
    calling enviar() then throws because it does not exist yet.
    """
    for state in ("load", "networkidle"):
        try:
            pg.wait_for_load_state(state, timeout=timeout)
        except Exception:
            pass


def _trigger_solicitud(pg, diag):
    """Ask for availability. The page's JS bundle does not always load, so fall
    back from the real button to the JS call to a bare form submit."""
    # Give the bundle a last chance to define enviar() before deciding.
    if diag.get("enviar") != "function":
        try:
            pg.wait_for_function("() => typeof enviar === 'function'", timeout=15000)
            diag["enviar"] = "function"
            log("enviar() showed up after waiting for the JS bundle")
        except Exception:
            log("enviar() still undefined after waiting; using fallbacks")

    attempts = []

    for btn in diag.get("buttons", []):
        label = f"{btn.get('value') or ''} {btn.get('text') or ''}".lower()
        if "solicitar" in label and btn.get("id"):
            attempts.append(("click #" + btn["id"], lambda i=btn["id"]: pg.click("#" + i)))
            break

    if diag.get("enviar") == "function":
        attempts.append(("enviar('solicitud')",
                         lambda: pg.evaluate("enviar('solicitud')")))

    attempts.append(("form submit", lambda: pg.evaluate(
        "() => { const f = document.forms[0]; if (!f) return false;"
        "  let m = f.querySelector('[name=method]');"
        "  if (m) m.value = 'solicitud'; f.submit(); return true; }")))

    for label, action in attempts:
        try:
            with pg.expect_navigation(timeout=45000):
                action()
            log(f"solicitud triggered via {label}")
            return True
        except Exception as e:
            log(f"solicitud via {label} failed: {type(e).__name__} {str(e)[:100]}")
    return False


def check_once(save_html=None):
    """Run one full check. Returns (status, detail)."""
    with sync_playwright() as p:
        browser = new_browser(p)
        ctx = browser.new_context(
            locale="es-ES",
            timezone_id="Europe/Madrid",
            user_agent=UA,
            viewport={"width": 1400, "height": 1000},
            ignore_https_errors=True,
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        )
        pg = ctx.new_page()
        pg.set_default_timeout(45000)
        try:
            # 1) front door. Deep-linking straight to acInfo works, and is what
            # the well-known bots do, but the WAF reads it as forced browsing
            # and starts rejecting the trámite page within a minute even from a
            # never-used IP. Walking the form is slower and stays welcome.
            pg.goto(f"{BASE}/{CATEGORY}/index",
                    wait_until="domcontentloaded", timeout=90000)
            settle(pg)
            sleep_ms(3000, 6000)
            guard(pg, "index page")

            # 2) choose the province the way the page offers it: the option
            # values are the citar URLs and the button navigates to them.
            wait_for(pg, "select[name=form]", "province select")
            province_url = f"/{CATEGORY}/citar?p={PROVINCE}&locale=es"
            try:
                pg.select_option("select[name=form]", province_url)
            except Exception:
                pg.select_option("select[name=form]",
                                 label=re.compile(PROVINCE_NAME, re.I))
            sleep_ms(1500, 3000)
            pg.click("#btnAceptar")
            pg.wait_for_load_state("domcontentloaded")
            settle(pg)
            sleep_ms(3000, 6000)
            guard(pg, "province page")

            # 3) pick office + trámite and submit that form too
            wait_for(pg, f"select[name='{TRAMITE_PARAM}']", "trámite select")
            try:
                pg.select_option("select[name=sede]", SEDE)
            except Exception:
                pass
            sleep_ms(1200, 2500)
            pg.select_option(f"select[name='{TRAMITE_PARAM}']", TRAMITE)
            sleep_ms(2000, 4000)
            pg.click("#btnAceptar")
            pg.wait_for_load_state("domcontentloaded")
            settle(pg)
            sleep_ms(3000, 6000)
            text = guard(pg, "tramite page")
            if "CITA PREVIA" not in text.upper():
                return ERROR, f"unexpected page: {text[:200]!r}"

            # 4) instructions page -> Entrar
            wait_for(pg, "#btnEntrar", "instructions page")
            sleep_ms(1500, 3000)
            pg.click("#btnEntrar")
            pg.wait_for_load_state("domcontentloaded")
            settle(pg)
            sleep_ms(3000, 5000)

            # 4) identity form (fields refuse paste -> type it)
            wait_for(pg, "#txtIdCitado", "identity form")
            pg.click("#txtIdCitado")
            pg.type("#txtIdCitado", NIE, delay=random.randint(80, 180))
            sleep_ms(900, 1800)
            pg.click("#txtDesCitado")
            pg.type("#txtDesCitado", FULL_NAME, delay=random.randint(80, 180))
            sleep_ms(2500, 4500)
            pg.click("#btnEnviar")
            pg.wait_for_load_state("domcontentloaded")
            settle(pg)
            sleep_ms(3000, 5000)

            # 5) identity confirmed -> ask for availability (read-only)
            wait_for(pg, "#btnConsultar", "identity confirmation")
            sleep_ms(1500, 3000)

            diag = pg.evaluate("""() => ({
                enviar: typeof enviar,
                buttons: [...document.querySelectorAll(
                    'input[type=button],input[type=submit],button')].map(e => ({
                        id: e.id, value: e.value,
                        onclick: (e.getAttribute('onclick') || '').slice(0, 70),
                        text: (e.innerText || '').trim().slice(0, 30)})),
                forms: [...document.forms].map(f => ({id: f.id, action: f.action}))
            })""")
            log(f"identity page controls: {diag}")

            log(f"identity page url: {pg.url}")
            if not _trigger_solicitud(pg, diag):
                return ERROR, f"could not trigger solicitud; controls={diag}"
            log(f"after solicitud url: {pg.url}")

            # The result page can be slow over the VPN, so poll for a verdict
            # instead of sleeping a fixed amount and hoping.
            deadline = time.time() + 60
            text, verdict = "", None
            while time.time() < deadline:
                text = guard(pg, "result page")
                verdict = classify(pg, text)
                if verdict:
                    break
                time.sleep(2)

            if save_html:
                open(save_html, "w").write(pg.content())

            log(f"result page ({pg.url}): {text[:500]!r}")

            if verdict == NO_SLOTS:
                return NO_SLOTS, "no slots"
            if verdict == AVAILABLE:
                return AVAILABLE, offices_on_offer(pg)
            try:
                log("stuck-page structure: " + str(pg.evaluate(
                    "() => ({url: location.href,"
                    " forms: [...document.forms].map(f => ({id: f.id, name: f.name,"
                    "   action: f.action, method: f.method,"
                    "   fields: [...f.elements].map(e => e.name + ':' + e.type)})),"
                    " scripts: [...document.scripts].map(s => s.src).filter(Boolean)})")))
            except Exception:
                pass
            return ERROR, f"unrecognised result page: {text[:300]!r}"
        except Blocked as e:
            return BLOCKED, str(e)
        except Exception as e:
            return ERROR, f"{type(e).__name__}: {str(e)[:250]}"
        finally:
            try:
                browser.close()
            except Exception:
                pass


def rotate_exit_ip():
    """Hop to a different Spanish exit node. Returns True if the tunnel came
    back up, in which case the caller can retry promptly instead of retreating.

    A blocked IP stays blocked for a long while, so sitting out the backoff
    burns most of the monitoring window; a fresh node recovers in seconds.
    """
    log("rotating VPN exit node after block")
    try:
        r = subprocess.run(ROTATE_CMD, shell=True, timeout=180,
                           capture_output=True, text=True)
        for line in (r.stdout or "").splitlines():
            log("  " + line)
        if r.returncode != 0:
            log(f"  rotation failed rc={r.returncode}: {(r.stderr or '')[:200]}")
            return False
        return True
    except Exception as e:
        log(f"  rotation error: {type(e).__name__} {str(e)[:150]}")
        return False


def notify_available(offices):
    url = f"{BASE}/{CATEGORY}/citar?p={PROVINCE}"
    where = ""
    if offices:
        where = "\n🏢 <b>Вільні офіси:</b>\n" + "\n".join(f"  • {o}" for o in offices) + "\n"
    send_telegram(
        "🚨 <b>Є СЛОТИ НА CITA PREVIA!</b>\n\n"
        f"📍 <b>{PROVINCE_NAME}</b>\n"
        f"📋 {TRAMITE_NAME}\n"
        f"👤 {FULL_NAME} — {NIE}\n"
        f"{where}\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%H:%M')} UTC — біжи зараз, "
        "слоти розбирають за хвилини\n\n"
        f"➡️ <a href=\"{url}\">Відкрити сайт</a>  (потрібен іспанський VPN)\n\n"
        "<i>Дату видно вже на сайті, після вибору офісу. Бот туди не заходить, "
        "щоб не займати слот — бронювання і SMS-код лишаються за тобою.</i>"
    )


def main():
    if not NIE or not FULL_NAME:
        log("FATAL: NIE / FULL_NAME not set")
        sys.exit(1)

    log(f"Monitoring {PROVINCE_NAME} (p={PROVINCE}) tramite={TRAMITE} "
        f"every ~{INTERVAL}s; loop={LOOP_MINUTES}min")

    deadline = time.time() + LOOP_MINUTES * 60 if LOOP_MINUTES else 0
    last_status = None
    last_notified = 0.0
    consecutive_errors = 0
    block_streak = 0
    rotations = 0

    while True:
        status, detail = check_once()

        if status == AVAILABLE:
            log(f"*** SLOTS AVAILABLE *** offices={detail}")
            now = time.time()
            if last_status != AVAILABLE or (now - last_notified) > REMIND_AFTER:
                notify_available(detail)
                last_notified = now
            consecutive_errors = block_streak = 0
        elif status == NO_SLOTS:
            log("no slots")
            consecutive_errors = block_streak = 0
        elif status == BLOCKED:
            block_streak += 1
            log(f"BLOCKED by WAF ({block_streak}): {detail}")
            # Rotating is cheap but not free: each hop opens a VPN session, and
            # 14 hops in 25 minutes got the account's logins refused outright.
            # Past a few fruitless hops the address is clearly not the problem,
            # so stop burning sessions and wait instead.
            if ROTATE_CMD and rotations < MAX_ROTATIONS and rotate_exit_ip():
                rotations += 1
                block_streak = 0
                last_status = status
                time.sleep(random.uniform(30, 60))
                continue
            if rotations >= MAX_ROTATIONS:
                log(f"{rotations} rotations without luck — the exit IP is not "
                    f"the problem; waiting it out instead")
        else:
            consecutive_errors += 1
            log(f"ERROR ({consecutive_errors}): {detail}")
            if consecutive_errors == 12:
                send_telegram(
                    "⚠️ <b>Cita-монітор: 12 помилок поспіль</b>\n\n"
                    f"<code>{detail[:300]}</code>\n\n"
                    "Можливо, сайт змінився або блокує запити."
                )

        last_status = status

        if not LOOP_MINUTES:
            log(f"RESULT: {status} tramite={TRAMITE} province={PROVINCE}")
            return 0 if status in (AVAILABLE, NO_SLOTS) else 1
        if time.time() >= deadline:
            log("loop window finished")
            return 0

        if block_streak:
            # Exponential retreat: 10, 20, 40, capped at 60 minutes. Hammering a
            # WAF that just said no is how you earn a longer ban.
            wait = min(600 * 2 ** (block_streak - 1), 3600)
            log(f"backing off {wait // 60} min after block")
        else:
            wait = INTERVAL + random.uniform(-20, 40)
        time.sleep(wait)


if __name__ == "__main__":
    sys.exit(main() or 0)
