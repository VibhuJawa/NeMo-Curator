#!/usr/bin/env bash

_NEBIUS_SSH_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_NEBIUS_SSH_WORKSPACE_DIR="$(cd "${_NEBIUS_SSH_LIB_DIR}/.." && pwd)"

nebius_ssh_host_candidates() {
  local host="$1"
  local user_prefix=""
  local bare_host="$host"
  local cached_host
  if [[ "$host" == *@* ]]; then
    user_prefix="${host%@*}@"
    bare_host="${host#*@}"
  fi

  nebius_emit_host_candidate() {
    local candidate="$1"
    if [[ "$candidate" == *@* ]]; then
      printf '%s\n' "$candidate"
    else
      printf '%s\n' "${user_prefix}${candidate}"
    fi
  }

  if [[ "${NEBIUS_SSH_PREFER_LAST_GOOD:-1}" != "0" && "$bare_host" == nb-hel-cs-001-* ]]; then
    cached_host="$(nebius_ssh_cached_host 2>/dev/null || true)"
    if [[ -n "$cached_host" ]]; then
      nebius_emit_host_candidate "$cached_host"
    fi
  fi

  nebius_emit_host_candidate "$bare_host"

  if [[ "$bare_host" == *.nvidia.com ]]; then
    nebius_emit_host_candidate "${bare_host%.nvidia.com}.cm.cluster"
  elif [[ "$bare_host" == *.cm.cluster ]]; then
    nebius_emit_host_candidate "${bare_host%.cm.cluster}.nvidia.com"
  fi

  case "$bare_host" in
    nb-hel-cs-001-*)
      nebius_emit_host_candidate "nb-hel-cs-001-vscode-01.nvidia.com"
      nebius_emit_host_candidate "nb-hel-cs-001-vscode-01.cm.cluster"
      nebius_emit_host_candidate "nb-hel-cs-001-vscode-02.nvidia.com"
      nebius_emit_host_candidate "nb-hel-cs-001-vscode-02.cm.cluster"
      nebius_emit_host_candidate "nb-hel-cs-001-login-02.nvidia.com"
      nebius_emit_host_candidate "nb-hel-cs-001-login-02.cm.cluster"
      nebius_emit_host_candidate "nb-hel-cs-001-login-01.nvidia.com"
      nebius_emit_host_candidate "nb-hel-cs-001-login-01.cm.cluster"
      nebius_emit_host_candidate "nb-hel-cs-001-dc-02.nvidia.com"
      nebius_emit_host_candidate "nb-hel-cs-001-dc-02.cm.cluster"
      nebius_emit_host_candidate "nb-hel-cs-001-dc-01.nvidia.com"
      nebius_emit_host_candidate "nb-hel-cs-001-dc-01.cm.cluster"
      ;;
  esac

  case "$bare_host" in
    nb-hel-cs-001-login-01*)
      nebius_emit_host_candidate "nb-hel-cs-001-vscode-01.nvidia.com"
      nebius_emit_host_candidate "nb-hel-cs-001-vscode-01.cm.cluster"
      ;;
    nb-hel-cs-001-vscode-01*)
      nebius_emit_host_candidate "nb-hel-cs-001-login-01.nvidia.com"
      nebius_emit_host_candidate "nb-hel-cs-001-login-01.cm.cluster"
      ;;
  esac

  if [[ -n "${NEBIUS_SSH_HOST_FALLBACKS:-}" ]]; then
    while IFS= read -r candidate; do
      [[ -n "$candidate" ]] || continue
      nebius_emit_host_candidate "$candidate"
    done < <(tr ',:' '\n' <<<"${NEBIUS_SSH_HOST_FALLBACKS}" | sed '/^$/d')
  fi
}

nebius_ssh_error_is_transient() {
  local error_file="$1"
  grep -Eqi 'Could not resolve hostname|Name or service not known|nodename nor servname provided|Temporary failure in name resolution|Connection timed out|Operation timed out' "$error_file"
}

nebius_ssh_control_dir() {
  printf '%s\n' "${NEBIUS_SSH_CONTROL_DIR:-${_NEBIUS_SSH_WORKSPACE_DIR}/.nebius_ssh_control}"
}

nebius_ssh_normalized_target() {
  local candidate="$1"
  local bare_host="$candidate"
  local user="${NEBIUS_SSH_USER:-${USER:-}}"

  if [[ "$candidate" == *@* ]]; then
    user="${candidate%@*}"
    bare_host="${candidate#*@}"
  fi

  if [[ -n "$user" ]]; then
    printf '%s@%s\n' "$user" "$bare_host"
  else
    printf '%s\n' "$bare_host"
  fi
}

nebius_ssh_control_path() {
  local candidate="$1"
  local control_dir
  local key
  control_dir="$(nebius_ssh_control_dir)"
  key="$(nebius_ssh_normalized_target "$candidate" | cksum | awk '{print $1 "_" $2}')"
  printf '%s/%s.sock\n' "$control_dir" "$key"
}

nebius_ssh_cache_file() {
  printf '%s/last_good_host\n' "$(nebius_ssh_control_dir)"
}

nebius_ssh_cached_host() {
  local cache_file
  cache_file="$(nebius_ssh_cache_file)"
  [[ -f "$cache_file" ]] || return 1
  sed -n '1p' "$cache_file"
}

nebius_ssh_cache_success() {
  local candidate="$1"
  local control_dir
  local cache_file
  control_dir="$(nebius_ssh_control_dir)"
  cache_file="$(nebius_ssh_cache_file)"
  mkdir -p "$control_dir"
  nebius_ssh_normalized_target "$candidate" >"$cache_file"
}

nebius_ssh_base_options() {
  local candidate="$1"
  local connect_timeout="$2"
  local control_dir
  local control_path

  printf '%s\n' \
    -o BatchMode=yes \
    -o ConnectTimeout="$connect_timeout" \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=2

  if [[ "${NEBIUS_SSH_CONTROL_MASTER:-1}" != "0" ]]; then
    control_dir="$(nebius_ssh_control_dir)"
    mkdir -p "$control_dir"
    control_path="$(nebius_ssh_control_path "$candidate")"
    printf '%s\n' \
      -o ControlMaster=auto \
      -o ControlPersist="${NEBIUS_SSH_CONTROL_PERSIST:-4h}" \
      -o ControlPath="$control_path"
  else
    # Be explicit so a user's ~/.ssh/config ControlMaster/ControlPath cannot
    # leak into Codex sandboxed runs and trip local socket permissions.
    printf '%s\n' \
      -o ControlMaster=no \
      -o ControlPath=none
  fi
}

nebius_ssh_command() {
  local host="$1"
  shift
  nebius_ssh_run "$host" "" "$@"
}

nebius_ssh_command_string() {
  local candidate="$1"
  local connect_timeout="${2:-${NEBIUS_SSH_CONNECT_TIMEOUT:-30}}"
  local opt
  local ssh_opts

  ssh_opts=("ssh")
  while IFS= read -r opt; do
    ssh_opts+=("$opt")
  done < <(nebius_ssh_base_options "$candidate" "$connect_timeout")

  printf '%q' "${ssh_opts[0]}"
  for opt in "${ssh_opts[@]:1}"; do
    printf ' %q' "$opt"
  done
  printf '\n'
}

nebius_resolve_ssh_host() {
  local host="$1"
  local attempts="${NEBIUS_SSH_ATTEMPTS:-3}"
  local retry_delay="${NEBIUS_SSH_RETRY_DELAY:-3}"
  local connect_timeout="${NEBIUS_SSH_CONNECT_TIMEOUT:-30}"
  local candidate
  local attempt
  local status=255
  local error_file
  local ssh_opts

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    for attempt in $(seq 1 "$attempts"); do
      error_file="$(mktemp "${TMPDIR:-/tmp}/nebius_ssh_resolve.XXXXXX")"
      ssh_opts=()
      while IFS= read -r opt; do
        ssh_opts+=("$opt")
      done < <(nebius_ssh_base_options "$candidate" "$connect_timeout")
      if ssh "${ssh_opts[@]}" "$candidate" "true" 2>"$error_file"; then
        status=0
      else
        status=$?
      fi
      if [[ "$status" -eq 0 ]]; then
        nebius_ssh_cache_success "$candidate"
        rm -f "$error_file"
        printf '%s\n' "$candidate"
        return 0
      fi

      cat "$error_file" >&2
      if [[ "$status" -ne 255 ]] || ! nebius_ssh_error_is_transient "$error_file"; then
        rm -f "$error_file"
        return "$status"
      fi
      rm -f "$error_file"

      if [[ "$attempt" -lt "$attempts" ]]; then
        sleep "$retry_delay"
      fi
    done
  done < <(nebius_ssh_host_candidates "$host" | awk '!seen[$0]++')

  return "$status"
}

nebius_resolve_rsync_host() {
  # Return a dc (data-copier) node for file transfers. DC nodes are much faster
  # than login/vscode nodes for bulk rsync/scp. Falls back to the given host if
  # it is already a dc node or not a Nebius cluster host.
  local host="$1"
  local user_prefix=""
  local bare_host="$host"
  if [[ "$host" == *@* ]]; then
    user_prefix="${host%@*}@"
    bare_host="${host#*@}"
  fi

  if [[ "$bare_host" == nb-hel-cs-001-dc-* ]]; then
    printf '%s\n' "$host"
    return 0
  fi

  if [[ "$bare_host" == nb-hel-cs-001-* ]]; then
    local dc_host="${NEBIUS_RSYNC_HOST:-nb-hel-cs-001-dc-01.nvidia.com}"
    printf '%s%s\n' "$user_prefix" "$dc_host"
    return 0
  fi

  printf '%s\n' "$host"
}

nebius_ssh_stdin() {
  local host="$1"
  shift

  local input_file
  input_file="$(mktemp "${TMPDIR:-/tmp}/nebius_ssh_stdin.XXXXXX")"
  cat >"$input_file"
  nebius_ssh_run "$host" "$input_file" "$@"
  local status=$?
  rm -f "$input_file"
  return "$status"
}

nebius_ssh_run() {
  local host="$1"
  local input_file="$2"
  shift 2

  local attempts="${NEBIUS_SSH_ATTEMPTS:-3}"
  local retry_delay="${NEBIUS_SSH_RETRY_DELAY:-3}"
  local connect_timeout="${NEBIUS_SSH_CONNECT_TIMEOUT:-30}"
  local candidate
  local attempt
  local status=255
  local error_file
  local ssh_opts

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    for attempt in $(seq 1 "$attempts"); do
      error_file="$(mktemp "${TMPDIR:-/tmp}/nebius_ssh.XXXXXX")"
      ssh_opts=()
      while IFS= read -r opt; do
        ssh_opts+=("$opt")
      done < <(nebius_ssh_base_options "$candidate" "$connect_timeout")
      if [[ -n "$input_file" ]]; then
        if ssh "${ssh_opts[@]}" "$candidate" "$@" <"$input_file" 2>"$error_file"; then
          status=0
        else
          status=$?
        fi
      else
        if ssh "${ssh_opts[@]}" "$candidate" "$@" 2>"$error_file"; then
          status=0
        else
          status=$?
        fi
      fi
      if [[ "$status" -eq 0 ]]; then
        nebius_ssh_cache_success "$candidate"
        rm -f "$error_file"
        return 0
      fi

      cat "$error_file" >&2
      if [[ "$status" -ne 255 ]] || ! nebius_ssh_error_is_transient "$error_file"; then
        rm -f "$error_file"
        return "$status"
      fi
      rm -f "$error_file"

      if [[ "$attempt" -lt "$attempts" ]]; then
        sleep "$retry_delay"
      fi
    done
  done < <(nebius_ssh_host_candidates "$host" | awk '!seen[$0]++')

  return "$status"
}
