#!/usr/bin/env python3
"""
Print the chat_id of everyone who has written to the bot.

A Telegram bot cannot message someone first, so every recipient must press
Start / send any message to the bot before their chat_id shows up here.

    TELEGRAM_TOKEN=123:AA... python get_chat_ids.py
"""
import os
import sys

import requests

token = os.environ.get("TELEGRAM_TOKEN") or (sys.argv[1] if len(sys.argv) > 1 else "")
if not token:
    sys.exit("Usage: TELEGRAM_TOKEN=<token> python get_chat_ids.py")

hook = requests.get(f"https://api.telegram.org/bot{token}/getWebhookInfo", timeout=30).json()
if hook.get("result", {}).get("url"):
    print(f"WARNING: a webhook is set ({hook['result']['url']}) — getUpdates will "
          f"stay empty until you delete it.\n")

r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30).json()
if not r.get("ok"):
    sys.exit(f"Telegram error: {r}")

seen = {}
for upd in r.get("result", []):
    msg = upd.get("message") or upd.get("edited_message") or {}
    chat = msg.get("chat")
    if chat:
        who = chat.get("username") or " ".join(
            filter(None, [chat.get("first_name"), chat.get("last_name")]))
        seen[chat["id"]] = who or "?"

if not seen:
    print("No messages yet. Open the bot in Telegram, press Start, then rerun.")
else:
    for cid, who in seen.items():
        print(f"{cid}\t{who}")
    print("\nCHAT_IDS=" + ",".join(str(c) for c in seen))
