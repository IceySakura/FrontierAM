#!/usr/bin/env bash
# dind-runc — runc wrapper for the inner dockerd inside the DinD sandbox.
#
# At `create` it rewrites the OCI spec:
#   1) bind-mounts the merged CA bundle (public roots + Squid CA) and the
#      squid-only CA into fixed in-container paths;
#   2) injects proxy / mirror / CA env into .process.env (respecting user-set
#      values via `docker run -e`);
#   3) when the image ships a shell, appends a `startContainer` OCI hook that
#      installs the squid CA into the container's own trust stores
#      (system store / certifi / java) — same effect as suanlicluster.py,
#      executed in the container mount ns after pivot_root so it never touches
#      the host.
# Every other runc subcommand is passed through untouched. Any injection
# failure is swallowed so it can never abort container creation.
#
# Runtime state files produced by docker-bootstrap.sh:
#   /var/run/dind-ca-bundle.crt  — public roots + Squid CA (CA env target src)
#   /var/run/dind-inject.env     — proxy + mirror env whitelist snapshot
#   /usr/local/share/ca-certificates/squid-proxy-ca.crt — squid-only CA (host)
#
# This script is packaged by the nix flake into
# /nix/runtime/nix-env/bin/dind-runc — it does NOT need runtime code
# generation. The real runc is resolved from non-wrapper nix store entries to
# avoid recursion after PATH and moby libexec/docker/runc are repointed here;
# jq still uses PATH with a fixed nix-env fallback.
set -euo pipefail

# Resolve the real runc binary. dind-runc is installed as "runc" in both the
# nix profile and moby's libexec/docker/runc, so we must not use
# `command -v runc` — it may resolve back to this wrapper. Instead, select a
# non-wrapper runc from the nix store. When BuildKit reaches us through moby's
# libexec/docker/runc path, prefer docker-runc; otherwise prefer standalone runc
# and fall back to docker-runc if standalone runc is not present in the closure.
RUNC_BIN=""
case "$0" in
    *moby*libexec/docker/runc*|*libexec/docker/runc*)
        for c in /nix/store/*-docker-runc-*/bin/runc; do
            [[ -x "$c" ]] && { RUNC_BIN="$c"; break; }
        done
        ;;
    *)
        for c in /nix/store/*-runc-*/bin/runc; do
            case "$c" in *docker-runc*|*dind-runc*) continue ;; esac
            [[ -x "$c" ]] && { RUNC_BIN="$c"; break; }
        done
        if [[ ! -x "$RUNC_BIN" ]]; then
            for c in /nix/store/*-docker-runc-*/bin/runc; do
                [[ -x "$c" ]] && { RUNC_BIN="$c"; break; }
            done
        fi
        ;;
esac
if [[ ! -x "$RUNC_BIN" ]]; then
    echo "dind-runc: FATAL: cannot locate real runc binary (called as $0)" >&2
    exit 1
fi

JQ_BIN="$(command -v jq 2>/dev/null || true)"
[[ -x "$JQ_BIN"   ]] || JQ_BIN=/nix/runtime/nix-env/bin/jq

CA=/var/run/dind-ca-bundle.crt        # public roots + Squid CA (from docker-bootstrap.sh)
DST=/etc/ssl/dind-proxy-ca.crt        # in-container mount of the FULL bundle (CA env target)
SQUID_SRC=/etc/ssl/dind-squid-ca.crt  # in-container mount of the squid-ONLY CA (hook input)
INJECT_ENV=/var/run/dind-inject.env   # proxy + mirror env whitelist snapshot
SQUID_CA="${SQUID_CA:-/usr/local/share/ca-certificates/squid-proxy-ca.crt}"

# CA install script run INSIDE the container by the startContainer hook (runs in
# the container mount ns after pivot_root, so it touches only the container's own
# filesystem — never the host). Complete port of suanlicluster.py's cert init:
#   1) per-distro system trust store  (Alpine / Debian / RHEL / fallback)
#   2) python certifi bundle in every reachable interpreter
#   3) java cacerts — iterate /usr/local/share/ca-certificates/* (not just squid),
#      so user-image-baked internal CAs get imported too, matching suanlicluster
#   4) npm cafile (npm ignores system store, needs explicit config)
#   5) maven settings.xml proxy injection (Maven 3.9+ HttpClient 5 doesn't read
#      JVM -Dhttps.proxyHost, needs settings.xml)
#   6) unified /etc/ssl/certs/ca-certificates.crt symlink for SSL_CERT_FILE
# Fully best-effort (exit 0) so a hook failure can never block the container.
CA_INSTALL_SCRIPT='export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}
CRT=/etc/ssl/dind-squid-ca.crt
[ -s "$CRT" ] || exit 0
M=$(sed -n 2p "$CRT" 2>/dev/null)
ensure(){ [ -f "$1" ] || return 0; if [ -n "$M" ] && grep -qF "$M" "$1" 2>/dev/null; then return 0; fi; cat "$CRT" >> "$1" 2>/dev/null || true; }
# 1) system trust store — per distro
if [ -f /etc/alpine-release ]; then
  mkdir -p /usr/local/share/ca-certificates 2>/dev/null || true
  cp "$CRT" /usr/local/share/ca-certificates/squid-proxy-ca.crt 2>/dev/null || true
  command -v update-ca-certificates >/dev/null 2>&1 && update-ca-certificates 2>/dev/null || true
  ensure /etc/ssl/certs/ca-certificates.crt
elif [ -f /etc/redhat-release ] || [ -f /etc/centos-release ] || [ -f /etc/fedora-release ]; then
  mkdir -p /etc/pki/ca-trust/source/anchors 2>/dev/null || true
  cp "$CRT" /etc/pki/ca-trust/source/anchors/squid-proxy-ca.crt 2>/dev/null || true
  command -v update-ca-trust >/dev/null 2>&1 && update-ca-trust 2>/dev/null || true
  ensure /etc/pki/tls/certs/ca-bundle.crt
elif [ -f /etc/debian_version ]; then
  mkdir -p /usr/local/share/ca-certificates 2>/dev/null || true
  cp "$CRT" /usr/local/share/ca-certificates/squid-proxy-ca.crt 2>/dev/null || true
  command -v update-ca-certificates >/dev/null 2>&1 && update-ca-certificates 2>/dev/null || true
  ensure /etc/ssl/certs/ca-certificates.crt
else
  ensure /etc/ssl/certs/ca-certificates.crt
  ensure /etc/pki/tls/certs/ca-bundle.crt
fi
# 2) python certifi bundle — match suanlicluster.py python list exactly
for py in python python3 python3.11 python3.12 python3.13 /usr/local/bin/python3 /usr/local/bin/python /usr/bin/python3 /usr/bin/python /opt/conda/bin/python /opt/miniconda3/bin/python /opt/anaconda3/bin/python /root/miniconda3/bin/python /root/anaconda3/bin/python; do
  command -v "$py" >/dev/null 2>&1 || continue
  P=$("$py" -c "import certifi; print(certifi.where())" 2>/dev/null) || continue
  [ -n "$P" ] && [ -f "$P" ] && ensure "$P"
done
# 3) java cacerts — iterate the whole /usr/local/share/ca-certificates dir like
#    suanlicluster.py: user-image-baked internal CAs also get imported. Uses
#    `keytool -cacerts` (JDK 9+); best-effort so older JDK / distroless-java is fine.
if command -v keytool >/dev/null 2>&1; then
  for cert in /usr/local/share/ca-certificates/*; do
    [ -f "$cert" ] || continue
    alias=$(basename "$cert" .crt)
    keytool -import -alias "$alias" -cacerts -storepass changeit -file "$cert" -noprompt 2>/dev/null || true
  done
fi
# 4) npm cafile — npm has its own trust config, ignores system store
if command -v npm >/dev/null 2>&1; then
  npm config set cafile /etc/ssl/certs/ca-certificates.crt 2>/dev/null || true
fi
# 5) maven settings.xml proxy — Maven 3.9+ HttpClient 5 doesnt read JVM
#    -Dhttps.proxyHost, must go through settings.xml. Two branches: no file →
#    create; file exists but no <proxies> → sed inject; else skip. Matches
#    suanlicluster.py behavior exactly (hardcoded hysandbox-proxy.woa.com:3080).
if command -v mvn >/dev/null 2>&1; then
  M2="${HOME:-/root}/.m2"
  mkdir -p "$M2" 2>/dev/null || true
  SETTINGS="$M2/settings.xml"
  if [ ! -f "$SETTINGS" ]; then
    cat > "$SETTINGS" <<MAVEN_EOF 2>/dev/null || true
<settings>
  <proxies>
    <proxy>
      <id>http-proxy</id>
      <active>true</active>
      <protocol>http</protocol>
      <host>hysandbox-proxy.woa.com</host>
      <port>3080</port>
    </proxy>
    <proxy>
      <id>https-proxy</id>
      <active>true</active>
      <protocol>https</protocol>
      <host>hysandbox-proxy.woa.com</host>
      <port>3080</port>
    </proxy>
  </proxies>
</settings>
MAVEN_EOF
  elif ! grep -q "<proxies>" "$SETTINGS" 2>/dev/null; then
    sed -i "/<\/settings>/i \\
  <proxies>\\
    <proxy>\\
      <id>http-proxy</id>\\
      <active>true</active>\\
      <protocol>http</protocol>\\
      <host>hysandbox-proxy.woa.com</host>\\
      <port>3080</port>\\
    </proxy>\\
    <proxy>\\
      <id>https-proxy</id>\\
      <active>true</active>\\
      <protocol>https</protocol>\\
      <host>hysandbox-proxy.woa.com</host>\\
      <port>3080</port>\\
    </proxy>\\
  </proxies>" "$SETTINGS" 2>/dev/null || true
  fi
fi
# 6) unified SSL_CERT_FILE path — ensure /etc/ssl/certs/ca-certificates.crt exists
#    on all distros (RHEL/CentOS put it under /etc/pki/tls/, SUSE under /etc/ssl/)
mkdir -p /etc/ssl/certs 2>/dev/null || true
if [ ! -f /etc/ssl/certs/ca-certificates.crt ]; then
  if [ -f /etc/pki/tls/certs/ca-bundle.crt ]; then
    ln -sf /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt 2>/dev/null || true
  elif [ -f /etc/ssl/ca-bundle.pem ]; then
    ln -sf /etc/ssl/ca-bundle.pem /etc/ssl/certs/ca-certificates.crt 2>/dev/null || true
  fi
fi
exit 0'

op=""; bundle=""; proc_file=""; args=("$@")
for ((i=0;i<${#args[@]};i++)); do
  case "${args[$i]}" in
    create)      op=create ;;
    run)         op=run ;;       # BuildKit's runc executor uses `runc run`, not `create`+`start`
    exec)        op="exec" ;;
    --bundle|-b) bundle="${args[$((i+1))]}" ;;
    --bundle=*)  bundle="${args[$i]#*=}" ;;
    --process)   proc_file="${args[$((i+1))]}" ;;
    --process=*) proc_file="${args[$i]#*=}" ;;
  esac
done

cfg="${bundle:+$bundle/config.json}"
if [[ ("$op" == create || "$op" == run) && -n "$cfg" && -f "$cfg" ]] && "$JQ_BIN" --version >/dev/null 2>&1; then
  extra='[]'
  [[ -f "$INJECT_ENV" ]] && extra="$("$JQ_BIN" -R -s -c 'split("\n")|map(select(length>0))' "$INJECT_ENV")"
  if [[ -s "$CA" ]]; then has_ca=true; else has_ca=false; fi
  ca_mount='[]'
  [[ "$has_ca" == true ]] && ca_mount="$("$JQ_BIN" -n --arg dst "$DST" --arg src "$CA" \
      '[{destination:$dst,source:$src,type:"bind",options:["rbind","ro"]}]')"

  # Mount the squid-only CA into the container and, when the image ships a shell,
  # inject a startContainer hook that installs it into the container's own trust
  # stores (system store / certifi / java) via CA_INSTALL_SCRIPT — same effect as
  # suanlicluster.py, executed in the container mount ns so it never touches the
  # host. Images without any shell (e.g. distroless) get NO hook (a missing
  # hook.path would fail container start); they still rely on the CA env above.
  squid_mount='[]'; hooks='{}'
  if [[ -s "$SQUID_CA" ]]; then
    squid_mount="$("$JQ_BIN" -n --arg dst "$SQUID_SRC" --arg src "$SQUID_CA" \
        '[{destination:$dst,source:$src,type:"bind",options:["rbind","ro"]}]')"
    rp="$("$JQ_BIN" -r '.root.path // "rootfs"' "$cfg" 2>/dev/null || echo rootfs)"
    case "$rp" in /*) rootfs="$rp" ;; *) rootfs="$bundle/$rp" ;; esac
    for s in /bin/sh /bin/bash /usr/bin/sh /usr/bin/bash; do
      if [[ -e "$rootfs$s" || -L "$rootfs$s" ]]; then
        hooks="$("$JQ_BIN" -n --arg script "$CA_INSTALL_SCRIPT" \
            '{startContainer:[{path:"/bin/sh",args:["/bin/sh","-c",$script],timeout:60}]}')"
        break
      fi
    done
  fi

  # Rewrite the OCI spec to inject proxy / CA / hook. Steps:
  #   1) append CA bind mounts (full bundle for CA env + squid-only CA for hook)
  #   2) inject proxy + mirror env, only for keys not already set by user (respect docker run -e)
  #      (includes CA env keys from outer when hook will also install squid CA into system bundle)
  #   3) CA env: only when NO startContainer hook — force-point to bind-mounted bundle
  #      (distroless / no-shell — system bundle won't get squid CA). When hook EXISTS,
  #      skip step 3: the outer's CA env values (passed through in step 2) point to
  #      system bundle paths that the hook installed squid CA into — same as outer.
  #   4) startContainer hook: install squid CA into system store/certifi/java in-container
  #      (only when image has a shell; append, don't clobber existing hooks)
  if "$JQ_BIN" \
      --argjson extra "$extra" \
      --argjson ca_mount "$ca_mount" \
      --argjson squid_mount "$squid_mount" \
      --argjson hooks "$hooks" \
      --argjson has_ca "$has_ca" \
      --arg dst "$DST" '
      def key: split("=")[0];
      .mounts = ((.mounts // []) + $ca_mount + $squid_mount)
      | (.process.env // []) as $env
      | ($env | map(key)) as $have
      | .process.env = ($env + ($extra | map(select((key as $k | ($have | index($k)) == null)))))
      | if $has_ca and ($hooks | has("startContainer") | not) then
          ["SSL_CERT_FILE","SSL_CERT_DIR","REQUESTS_CA_BUNDLE","CURL_CA_BUNDLE","NODE_EXTRA_CA_CERTS","GIT_SSL_CAINFO","PIP_CERT"] as $ca_keys
          | .process.env = (.process.env | map(select((key as $k | ($ca_keys | index($k)) == null))))
          | .process.env += ["SSL_CERT_FILE="+$dst, "SSL_CERT_DIR=", "REQUESTS_CA_BUNDLE="+$dst,
                             "CURL_CA_BUNDLE="+$dst, "NODE_EXTRA_CA_CERTS="+$dst,
                             "GIT_SSL_CAINFO="+$dst, "PIP_CERT="+$dst]
        else . end
      | if ($hooks | has("startContainer")) then
          .hooks = (.hooks // {})
          | .hooks.startContainer = ((.hooks.startContainer // []) + $hooks.startContainer)
        else . end
    ' "$cfg" > "$cfg.tmp" 2>/dev/null; then
    mv "$cfg.tmp" "$cfg"
  else
    rm -f "$cfg.tmp"
  fi
fi

# Intercept `exec` subcommand: inject proxy + CA env into the exec process spec
# (passed via --process <file>). Mounts and hooks from create already persist in
# the container's mount namespace, so exec only needs env injection. Without this,
# `docker exec` processes would have no proxy/CA env — breaking debugging inside
# running containers.
if [[ "$op" == exec && -n "$proc_file" && -f "$proc_file" ]] && "$JQ_BIN" --version >/dev/null 2>&1; then
  extra='[]'
  [[ -f "$INJECT_ENV" ]] && extra="$("$JQ_BIN" -R -s -c 'split("\n")|map(select(length>0))' "$INJECT_ENV")"
  if [[ -s "$CA" ]]; then has_ca=true; else has_ca=false; fi

  CA_TARGET="$DST"
  if [[ -f "$INJECT_ENV" ]]; then
    outer_ssl=$(grep '^SSL_CERT_FILE=' "$INJECT_ENV" | head -1 | cut -d= -f2-)
    [[ -n "$outer_ssl" ]] && CA_TARGET="$outer_ssl"
  fi

  if "$JQ_BIN" \
      --argjson extra "$extra" \
      --argjson has_ca "$has_ca" \
      --arg ca_target "$CA_TARGET" '
      def key: split("=")[0];
      (.env // []) as $env
      | ($env | map(key)) as $have
      | .env = ($env + ($extra | map(select((key as $k | ($have | index($k)) == null)))))
      | if $has_ca then
          ["SSL_CERT_FILE","SSL_CERT_DIR","REQUESTS_CA_BUNDLE","CURL_CA_BUNDLE","NODE_EXTRA_CA_CERTS","GIT_SSL_CAINFO","PIP_CERT"] as $ca_keys
          | .env = (.env | map(select((key as $k | ($ca_keys | index($k)) == null))))
          | .env += ["SSL_CERT_FILE="+$ca_target, "SSL_CERT_DIR=", "REQUESTS_CA_BUNDLE="+$ca_target,
                     "CURL_CA_BUNDLE="+$ca_target, "NODE_EXTRA_CA_CERTS="+$ca_target,
                     "GIT_SSL_CAINFO="+$ca_target, "PIP_CERT="+$ca_target]
        else . end
    ' "$proc_file" > "$proc_file.tmp" 2>/dev/null; then
    mv "$proc_file.tmp" "$proc_file"
  else
    rm -f "$proc_file.tmp"
  fi
fi

exec "$RUNC_BIN" "$@"

