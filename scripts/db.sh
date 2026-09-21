#!/bin/bash
# Convenience wrapper for the project-local Postgres instance (not a brew
# service -- lives at .pgdata, port 5433, separate from any system Postgres).
set -e
cd "$(dirname "$0")/.."
PGBIN=/opt/homebrew/opt/postgresql@17/bin

case "$1" in
  start)
    "$PGBIN/pg_ctl" -D .pgdata -l .pgdata/logfile -o "-p 5433 -k /tmp" start
    ;;
  stop)
    "$PGBIN/pg_ctl" -D .pgdata stop
    ;;
  status)
    "$PGBIN/pg_ctl" -D .pgdata status
    ;;
  psql)
    "$PGBIN/psql" -h 127.0.0.1 -p 5433 -U postgres -d leetcode_tutor
    ;;
  *)
    echo "Usage: scripts/db.sh {start|stop|status|psql}"
    exit 1
    ;;
esac
