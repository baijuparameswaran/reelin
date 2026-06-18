#!/usr/bin/env bash
# Install weekly + monthly cron jobs that refresh the agents' local models.
#
# Weekly  : Mondays 03:30 — pull preferred models + smoke test.
# Monthly : 1st of month 04:00 — also pull fallbacks (--all).
#
# NOTE (WSL): WSL2 does not run cron unless enabled. If `crontab` jobs don't
# fire, either enable systemd + cron in /etc/wsl.conf, run `sudo service cron
# start` at login, OR schedule `wsl -d <distro> -- bash -lc '.../update-models.sh'`
# from Windows Task Scheduler. On-demand `make update` always works regardless.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${ROOT}/scripts/update-models.sh"
WEEKLY="30 3 * * 1 ${SCRIPT} >> ${ROOT}/scripts/model-updates.log 2>&1"
MONTHLY="0 4 1 * * ${SCRIPT} --all >> ${ROOT}/scripts/model-updates.log 2>&1"

# Preserve existing crontab, drop any prior reel entries, add the two fresh ones.
current="$(crontab -l 2>/dev/null | grep -v 'update-models.sh' || true)"
printf '%s\n%s # reel-weekly\n%s # reel-monthly\n' "$current" "$WEEKLY" "$MONTHLY" \
  | sed '/^$/d' | crontab -

echo "Installed cron jobs:"
crontab -l | grep 'update-models.sh'
if ! pgrep -x cron >/dev/null 2>&1; then
  echo
  echo "⚠  cron does not appear to be running in this WSL session."
  echo "   Start it with:  sudo service cron start   (see notes in this script)."
fi
