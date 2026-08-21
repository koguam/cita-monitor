#!/bin/bash
# Reconnect the tunnel through a different Spanish exit node.
#
# The site rejects whole /24 blocks and a rejected block stays rejected for a
# long while, so waiting it out wastes most of the monitoring window. Hopping
# to a fresh block recovers in seconds instead.
set -u
CONF=/etc/openvpn/es.conf
[ -f "$CONF" ] || { echo "rotate: $CONF missing"; exit 1; }

# Remember the /24 we are leaving so the next pick avoids it rather than
# hopping to a neighbour that is blocked by the same rule.
BLOCKED_FILE="${BLOCKED_FILE:-/tmp/cita_blocked_subnets.txt}"
export BLOCKED_FILE
OLD_IP=$(awk '/^remote /{print $2; exit}' "$CONF")
if [ -n "${OLD_IP:-}" ]; then
  echo "${OLD_IP%.*}" >> "$BLOCKED_FILE"
  echo "rotate: marking ${OLD_IP%.*}.0/24 as burned"
fi

sudo pkill -f "openvpn --config $CONF" 2>/dev/null || true
sleep 2

exec bash connect_vpn.sh
