#!/usr/bin/env bash
# Run the official-wheel MP control and reject cache-lifecycle errors.
set -euo pipefail

_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_engine_log="${_root}/artifacts/official_lmcache_mp_engine.log"
_server_log="${_root}/artifacts/official_lmcache_mp_server.log"
_host="127.0.0.1"
_port="15555"
_http_port="18080"
mkdir -p "$(dirname "${_engine_log}")"

# shellcheck disable=SC1091
source "${_root}/scripts/activate_clean_lmcache.sh"

lmcache server \
  --host "${_host}" \
  --port "${_port}" \
  --http-host "${_host}" \
  --http-port "${_http_port}" \
  --l1-size-gb 2 \
  --eviction-policy LRU \
  --chunk-size 256 \
  >"${_server_log}" 2>&1 &
_server_pid=$!

cleanup() {
  if kill -0 "${_server_pid}" 2>/dev/null; then
    kill -TERM "${_server_pid}" 2>/dev/null || true
    wait "${_server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

_ready=0
for _attempt in $(seq 1 30); do
  if ! kill -0 "${_server_pid}" 2>/dev/null; then
    echo "LMCache MP server exited during startup; see ${_server_log}" >&2
    exit 1
  fi
  if grep -Eq "Started server process|Application startup complete" \
    "${_server_log}"; then
    _ready=1
    break
  fi
  sleep 1
done

if [[ "${_ready}" -ne 1 ]]; then
  echo "LMCache MP server did not become ready; see ${_server_log}" >&2
  exit 1
fi

python "${_root}/scripts/smoke_clean_lmcache.py" \
  --lmcache-host "${_host}" \
  --lmcache-port "${_port}" \
  "$@" 2>&1 | tee "${_engine_log}"

cleanup
trap - EXIT INT TERM

if grep -Eq "Double unpin|Pin count .* is negative|Traceback" \
  "${_engine_log}" "${_server_log}"; then
  echo "Official LMCache MP control failed lifecycle checks." >&2
  exit 1
fi

if ! grep -Eq "Retrieved .*1792|LMCache hit tokens: 1792" \
  "${_engine_log}" "${_server_log}"; then
  echo "Official LMCache MP control did not prove a 1792-token cache hit." >&2
  exit 1
fi

echo "Official LMCache control passed without lifecycle errors."
