#!/bin/bash
# Reconnect the tunnel through a different Spanish exit node.
#
# The WAF blocks by IP, and a blocked IP stays blocked for a long while, so
# waiting it out wastes most of the monitoring window. Hopping to a fresh
# NordVPN server recovers in about twenty seconds instead.
set -u
SITE_NET="${SITE_NET:-185.73.174.0/24}"
CONF=/etc/openvpn/es.conf

[ -f "$CONF" ] || { echo "rotate: $CONF missing"; exit 1; }

sudo pkill -f "openvpn --config $CONF" 2>/dev/null || true
sleep 2

PORT=$(awk '/^remote /{print $3; exit}' "$CONF")
PICK=$(python3 pick_server.py) || PICK=""
if [ -n "$PICK" ]; then
  IP=${PICK% *}
  HOST=${PICK#* }
  sudo sed -i "s#^remote .*#remote $IP ${PORT:-1194}#" "$CONF"
  sudo sed -i "s#^verify-x509-name .*#verify-x509-name CN=$HOST#" "$CONF"
  echo "rotate: now via $HOST ($IP)"
else
  echo "rotate: server list unavailable, reconnecting to the same node"
fi

sudo openvpn --config "$CONF" --daemon --log /tmp/ovpn.log
for _ in $(seq 1 30); do
  ip route | grep -q "${SITE_NET%/*}" && break
  sleep 2
done
if ! ip route | grep -q "${SITE_NET%/*}"; then
  echo "rotate: route did not come back"
  sudo tail -20 /tmp/ovpn.log
  exit 1
fi
echo "rotate: tunnel back up"
