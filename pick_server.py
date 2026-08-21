#!/usr/bin/env python3
"""
Print "<ip> <hostname>" for a Spanish NordVPN OpenVPN/UDP server.

NordVPN uses one CA and one tls-crypt key across its whole fleet, so a config
downloaded for one server works against any other once `remote` and
`verify-x509-name` are repointed.

Two things this deliberately does NOT do:

* It does not use /servers/recommendations. That endpoint returns ~15 hosts
  clustered in 187.13.x.x - the largest and most heavily used Spanish block,
  and the one the cita site was observed rejecting outright. The full list has
  121 servers across 23 /24 blocks.
* It does not pick uniformly across servers, which would just land in the big
  blocks again. It picks a /24 first, then a host inside it, so a block with
  three servers is as likely as one with thirty-nine.

Subnets already seen to be blocked are recorded in BLOCKED_FILE by
rotate_vpn.sh and skipped here. Prints nothing on failure, in which case the
caller keeps whatever server the config already names.
"""
import json
import os
import random
import sys
import urllib.request

API = ("https://api.nordvpn.com/v1/servers"
       "?filters%5Bcountry_id%5D=202&limit=500")          # 202 = Spain
BLOCKED_FILE = os.environ.get("BLOCKED_FILE", "/tmp/cita_blocked_subnets.txt")


def subnet24(ip):
    return ".".join(ip.split(".")[:3])


def supports_openvpn_udp(server):
    techs = server.get("technologies")
    if not techs:
        return True                      # unknown shape: do not exclude it
    return any(t.get("identifier") == "openvpn_udp" for t in techs)


def main():
    try:
        with urllib.request.urlopen(API, timeout=25) as r:
            servers = json.load(r)
    except Exception:
        return

    blocked = set()
    try:
        with open(BLOCKED_FILE) as fh:
            blocked = {line.strip() for line in fh if line.strip()}
    except OSError:
        pass

    by_subnet = {}
    for s in servers:
        ip, host = s.get("station"), s.get("hostname")
        if not ip or not host or not supports_openvpn_udp(s):
            continue
        by_subnet.setdefault(subnet24(ip), []).append((ip, host))

    fresh = {k: v for k, v in by_subnet.items() if k not in blocked}
    pool = fresh or by_subnet             # everything burned: start over
    if not pool:
        return

    ip, host = random.choice(pool[random.choice(list(pool))])
    print(ip, host, file=sys.stdout)


if __name__ == "__main__":
    main()
