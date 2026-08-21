#!/usr/bin/env python3
"""
Print "<ip> <hostname>" for a random Spanish NordVPN OpenVPN/UDP server.

NordVPN uses one CA and one tls-crypt key across its whole fleet, so a config
downloaded for one server works against any other once `remote` and
`verify-x509-name` are repointed. Rotating the exit IP between runs keeps the
monitor from hammering the cita site from a single address.

Prints nothing on failure — the caller then keeps the config's own server.
"""
import json
import random
import sys
import urllib.request

API = ("https://api.nordvpn.com/v1/servers/recommendations"
       "?filters%5Bcountry_id%5D=202"                       # 202 = Spain
       "&filters%5Bservers_technologies%5D%5Bidentifier%5D=openvpn_udp"
       "&limit=15")

try:
    with urllib.request.urlopen(API, timeout=20) as r:
        servers = json.load(r)
    usable = [s for s in servers if s.get("station") and s.get("hostname")]
    if not usable:
        sys.exit(0)
    # Prefer the lighter half of the list, then pick at random within it, so
    # we neither stampede one server nor always land on the same "best" one.
    usable.sort(key=lambda s: s.get("load", 100))
    pick = random.choice(usable[:max(1, len(usable) // 2)])
    print(pick["station"], pick["hostname"])
except Exception:
    sys.exit(0)
