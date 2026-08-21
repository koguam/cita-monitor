#!/bin/bash
# Bring the split tunnel up, trying several Spanish exit nodes.
#
# One attempt is not enough in practice: a node can refuse the login outright
# (AUTH_FAILED, seen when too many sessions are open on the account) or accept
# it and never install a route. Either way the fix is the same - try another
# node - so this keeps going until one works.
#
# Expects /etc/openvpn/es.conf and /etc/openvpn/creds to exist already.
set -u
SITE_NET="${SITE_NET:-185.73.174.0/24}"
CONF=/etc/openvpn/es.conf
LOG=/tmp/ovpn.log
ATTEMPTS="${ATTEMPTS:-5}"

[ -f "$CONF" ] || { echo "connect: $CONF missing"; exit 1; }
PORT=$(awk '/^remote /{print $3; exit}' "$CONF")
PORT=${PORT:-1194}

for attempt in $(seq 1 "$ATTEMPTS"); do
  PICK=$(python3 pick_server.py) || PICK=""
  if [ -n "$PICK" ]; then
    IP=${PICK% *}
    HOST=${PICK#* }
    sudo sed -i "s#^remote .*#remote $IP $PORT#" "$CONF"
    sudo sed -i "s#^verify-x509-name .*#verify-x509-name CN=$HOST#" "$CONF"
    echo "connect: attempt $attempt via $HOST ($IP)"
  else
    echo "connect: attempt $attempt via the config's own node"
  fi

  sudo rm -f "$LOG"
  sudo openvpn --config "$CONF" --daemon --log "$LOG"

  ok=""
  for _ in $(seq 1 25); do
    if ip route | grep -q "${SITE_NET%/*}"; then ok=1; break; fi
    if sudo grep -q "AUTH_FAILED" "$LOG" 2>/dev/null; then break; fi
    sleep 2
  done

  if [ -n "$ok" ]; then
    echo "connect: tunnel up via ${HOST:-config node}"
    exit 0
  fi

  if sudo grep -q "AUTH_FAILED" "$LOG" 2>/dev/null; then
    echo "connect: login rejected by ${HOST:-node}, trying another"
  else
    echo "connect: no route via ${HOST:-node}, trying another"
  fi
  sudo pkill -f "openvpn --config $CONF" 2>/dev/null || true
  sleep 3
done

echo "connect: every attempt failed"
if sudo grep -q "AUTH_FAILED" "$LOG" 2>/dev/null; then
  echo "::error::VPN login rejected on every node. Either the service"
  echo "::error::credentials are wrong, or the account has too many open"
  echo "::error::sessions - NordVPN allows a limited number at once."
fi
sudo tail -20 "$LOG" 2>/dev/null || true
exit 1
