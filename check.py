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
            # 1) province page (also clears the F5 JS challenge)
            pg.goto(f"{BASE}/{CATEGORY}/citar?p={PROVINCE}",
                    wait_until="domcontentloaded", timeout=90000)
            settle(pg)
            sleep_ms(4000, 7000)
            guard(pg, "province page")

            # 2) jump straight to the trámite (skips the select/submit dance)
            pg.goto(f"{BASE}/{CATEGORY}/acInfo?{TRAMITE_PARAM}={TRAMITE}",
                    wait_until="domcontentloaded", timeout=90000)
            settle(pg)
            sleep_ms(4000, 7000)
            text = guard(pg, "tramite page")
            if "CITA PREVIA" not in text.upper():
                return ERROR, f"unexpected page: {text[:200]!r}"

            # 3) instructions page -> Entrar
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
            text = ""
            while time.time() < deadline:
                text = guard(pg, "result page")
                if ("En este momento no hay citas disponibles" in text
                        or "Seleccione la oficina donde solicitar la cita" in text
                        or "Seleccione una de las siguientes citas disponibles" in text
                        or "DISPONE DE 5 MINUTOS" in text):
                    break
                time.sleep(2)

            if save_html:
                open(save_html, "w").write(pg.content())

            if "En este momento no hay citas disponibles" in text:
                return NO_SLOTS, "no slots"
            if ("Seleccione la oficina donde solicitar la cita" in text
                    or "Seleccione una de las siguientes citas disponibles" in text
                    or "DISPONE DE 5 MINUTOS" in text):
                return AVAILABLE, text[:600]
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


def notify_available(detail):
    url = f"{BASE}/{CATEGORY}/citar?p={PROVINCE}"
    send_telegram(
        "🚨 <b>Є СЛОТИ НА CITA PREVIA!</b>\n\n"
        f"📍 <b>{PROVINCE_NAME}</b>\n"
        f"📋 {TRAMITE_NAME}\n"
        f"👤 {FULL_NAME} — {NIE}\n\n"
        f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC\n\n"
        f"➡️ <a href=\"{url}\">Відкрити сайт і записатись</a>\n\n"
        "<i>Бот тільки перевіряє наявність — бронювати треба вручну, "
        "підтвердження приходить SMS-кодом.</i>"
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

    while True:
        status, detail = check_once()

        if status == AVAILABLE:
            log(f"*** SLOTS AVAILABLE *** {detail[:120]}")
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
