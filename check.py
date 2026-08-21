#!/usr/bin/env python3
"""
Cita Previa Extranjería monitor -> Telegram.

Watches the Spanish Sede Electrónica for free appointment slots for a single
trámite in a single province and pushes a Telegram message when slots appear.

It only ever READS availability. It never books, never picks a slot and never
submits contact details -- booking stays a manual step for the human.
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

# Optional Spanish proxy, in case the runner's IP is geo-blocked.
PROXY_SERVER = os.environ.get("PROXY_SERVER", "")
PROXY_USER = os.environ.get("PROXY_USER", "")
PROXY_PASS = os.environ.get("PROXY_PASS", "")

INTERVAL = int(os.environ.get("INTERVAL_SECONDS", "300"))       # between checks
REMIND_AFTER = int(os.environ.get("REMIND_AFTER_SECONDS", "1800"))  # re-ping while slots hold
LOOP_MINUTES = int(os.environ.get("LOOP_MINUTES", "0"))         # 0 = single check

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# Result codes
AVAILABLE, NO_SLOTS, ERROR = "AVAILABLE", "NO_SLOTS", "ERROR"


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
    """Real Chrome. The default Playwright headless shell is blocked by the
    site's FortiGate IPS; the real Chrome build passes."""
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
            sleep_ms(3000, 5000)

            # 2) jump straight to the trámite (skips the select/submit dance)
            pg.goto(f"{BASE}/{CATEGORY}/acInfo?{TRAMITE_PARAM}={TRAMITE}",
                    wait_until="domcontentloaded", timeout=90000)
            sleep_ms(3000, 5000)

            body = pg.inner_text("body")
            if "FortiGate" in body or "Request Rejected" in body:
                return ERROR, "blocked by WAF/IPS"
            if "CITA PREVIA" not in body.upper():
                return ERROR, f"unexpected page: {body[:200]!r}"

            # 3) instructions page -> Entrar
            pg.wait_for_selector("#btnEntrar", timeout=30000)
            sleep_ms(1000, 2000)
            pg.click("#btnEntrar")
            pg.wait_for_load_state("domcontentloaded")
            sleep_ms(2000, 4000)

            # 4) identity form (fields refuse paste -> type it)
            pg.wait_for_selector("#txtIdCitado", timeout=30000)
            pg.click("#txtIdCitado")
            pg.type("#txtIdCitado", NIE, delay=random.randint(60, 140))
            sleep_ms(700, 1500)
            pg.click("#txtDesCitado")
            pg.type("#txtDesCitado", FULL_NAME, delay=random.randint(60, 140))
            sleep_ms(2000, 4000)
            pg.click("#btnEnviar")
            pg.wait_for_load_state("domcontentloaded")
            sleep_ms(2000, 4000)

            # 5) identity confirmed -> ask for availability (read-only)
            pg.wait_for_selector("#btnConsultar", timeout=30000)
            sleep_ms(1000, 2000)
            try:
                with pg.expect_navigation(timeout=60000):
                    pg.evaluate("enviar('solicitud')")
            except Exception:
                pass  # the poll below is the real check

            # The result page can be slow over the VPN, so poll for a verdict
            # instead of sleeping a fixed amount and hoping.
            deadline = time.time() + 60
            body = ""
            while time.time() < deadline:
                try:
                    body = pg.inner_text("body")
                except Exception:
                    time.sleep(1)
                    continue
                if ("En este momento no hay citas disponibles" in body
                        or "Seleccione la oficina donde solicitar la cita" in body
                        or "Seleccione una de las siguientes citas disponibles" in body
                        or "DISPONE DE 5 MINUTOS" in body):
                    break
                time.sleep(2)

            if save_html:
                open(save_html, "w").write(pg.content())

            if "En este momento no hay citas disponibles" in body:
                return NO_SLOTS, "no slots"
            if ("Seleccione la oficina donde solicitar la cita" in body
                    or "Seleccione una de las siguientes citas disponibles" in body
                    or "DISPONE DE 5 MINUTOS" in body):
                return AVAILABLE, body[:600]
            return ERROR, f"unrecognised result page: {body[:300]!r}"
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
        "🚨 <b>ЄСТЬ СЛОТИ НА CITA PREVIA!</b>\n\n"
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

    while True:
        status, detail = check_once()

        if status == AVAILABLE:
            log(f"*** SLOTS AVAILABLE *** {detail[:120]}")
            now = time.time()
            if last_status != AVAILABLE or (now - last_notified) > REMIND_AFTER:
                notify_available(detail)
                last_notified = now
            consecutive_errors = 0
        elif status == NO_SLOTS:
            log("no slots")
            consecutive_errors = 0
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
            return 0 if status != ERROR else 1
        if time.time() >= deadline:
            log("loop window finished")
            return 0

        time.sleep(INTERVAL + random.uniform(-20, 40))


if __name__ == "__main__":
    sys.exit(main() or 0)
