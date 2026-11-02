{
  description = "nexus runtime";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }: {
    # Docker on Apple Silicon builds Linux ARM64 images.
    packages = nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      fontsConf = pkgs.makeFontsConf {
        fontDirectories = [
          pkgs.noto-fonts
          pkgs.noto-fonts-cjk-sans
          pkgs.noto-fonts-emoji
        ];
      };
      # Patched bashInteractive carrying the bash-capture mechanism:
      #   - cmd_wrap.o compiled in + linked with --wrap=execute_command
      #     (intercepts each command's execution, captures stdout/stderr/exit)
      #   - readline_wrap.o compiled in + linked with --wrap=readline
      #     (maintains batch/command_id counters, writes initial status.json)
      # The mechanism is gated by NEXUS_CAPTURE + NEXUS_CAPTURE_DIR env vars, so
      # this bash behaves as plain bash when they are unset.
      #
      # C sources live at deployment/runtime/bash/ (cmd_wrap.c,
      # readline_wrap.c, bc_common.c/h), read from ./bash relative to this
      # flake (the Dockerfile COPYs deployment/runtime into the build context).
      nexus_capture = pkgs.bashInteractive.overrideAttrs (old: {
        pname = "nexus-bash-capture";

        cmdWrapSrc  = ./bash/cmd_wrap.c;
        readlineWrapSrc = ./bash/readline_wrap.c;
        bcCommonSrc = ./bash/bc_common.c;
        bcCommonHdr = ./bash/bc_common.h;

        # NIX_LDFLAGS is passed straight to ld by cc-wrapper, so use bare ld
        # syntax (--wrap=...), NOT the gcc-driver -Wl, prefix. This wrap is a
        # no-op for links that don't reference execute_command (safe globally).
        # glibc>=2.34 folds pthread into libc, so no -lpthread needed.
        NIX_LDFLAGS = (old.NIX_LDFLAGS or "") + " --wrap=execute_command --wrap=readline";

        # After configure, before build: compile the capture wrappers and inject
        # them into the top-level bash OBJECTS line (only the bash binary links
        # the full OBJECTS, so these objects land only in bash, not helper tools).
        postConfigure = (old.postConfigure or "") + ''
          echo "[nexus-capture] staging bc_common.h"
          cp "$bcCommonHdr" ./bc_common.h
          echo "[nexus-capture] compiling bc_common.o"
          $CC $NIX_CFLAGS_COMPILE -I. -Iinclude -fPIC -c "$bcCommonSrc" -o bc_common.o
          echo "[nexus-capture] compiling cmd_wrap.o"
          $CC $NIX_CFLAGS_COMPILE -I. -Iinclude -fPIC -c "$cmdWrapSrc" -o cmd_wrap.o
          echo "[nexus-capture] compiling readline_wrap.o"
          $CC $NIX_CFLAGS_COMPILE -I. -Iinclude -fPIC -c "$readlineWrapSrc" -o readline_wrap.o
          echo "[nexus-capture] injecting bc_common.o + cmd_wrap.o + readline_wrap.o into top-level OBJECTS"
          sed -i -E 's/^OBJECTS([[:space:]]*)=([[:space:]]*)/OBJECTS\1=\2bc_common.o cmd_wrap.o readline_wrap.o /' Makefile
          grep -qE '^OBJECTS[[:space:]]*=[[:space:]]*bc_common\.o cmd_wrap\.o readline_wrap\.o ' Makefile \
            || { echo "[nexus-capture] ERROR: capture objects not injected"; exit 1; }
        '';
      });
      docker_bootstrap = pkgs.writeShellScriptBin "docker-bootstrap.sh" (builtins.readFile ./docker-bootstrap.sh);
      # dind-runc: runc wrapper packaged as a nix-provided binary so its path is
      # stable at /nix/runtime/nix-env/bin/dind-runc (the runtime image's fixed
      # nix-env root). docker-bootstrap.sh only points daemon.json at this path
      # and no longer emits the wrapper at runtime. The wrapper is fully static
      # — RUNC_BIN/JQ_BIN fall back to /nix/runtime/nix-env/bin/{runc,jq}.
      dind_runc = pkgs.writeShellScriptBin "dind-runc" (builtins.readFile ./dind-runc.sh);
      # Override bin/runc in the nix profile so PATH-based lookups (newer
      # BuildKit/containerd paths and manual `runc`) also route through dind-runc.
      # Docker 27 BuildKit additionally resolves runc through moby's libexec path;
      # Dockerfile.runtime repoints that store entry after copying the nix closure.
      dind_runc_override = pkgs.hiPrio (pkgs.runCommand "dind-runc-override" {} ''
        mkdir -p $out/bin
        ln -s ${dind_runc}/bin/dind-runc $out/bin/runc
      '');
      # Link docker (moby) against iptables-legacy. Under gVisor the nft backend is
      # unavailable, so the iptables the docker/libnetwork tooling shells out to must
      # be the legacy one. This is what makes inter-container service-name DNS
      # (the 127.0.0.11 embedded resolver) work inside the sandbox.
      dockerLegacy = pkgs.docker.override { iptables = pkgs.iptables-legacy; };
    in {
        # Standalone output so the bash+.so can be built/validated without the full env.
        nexus_capture = nexus_capture;

        default = pkgs.buildEnv {
          name = "nexus-runtime-lite";
          # nexus_capture provides bin/bash with cmd_wrap.o/readline_wrap.o
          # linked in via --wrap.
          # It is listed FIRST so its bin/bash wins over any
          # other package's bash under ignoreCollisions. nixpkgs `bash` is
          # intentionally NOT in this list.
          paths = [
            nexus_capture
          ] ++ (with pkgs; [
            tini
            uv
            python310
            python311
            python312
            python313
            python314

            # Docker-in-Docker runtime support
            dockerLegacy           # dockerd/docker CLI linked against iptables-legacy
            docker-buildx
            docker-compose
            docker_bootstrap
            dind_runc
            dind_runc_override
            iproute2
            iptables-legacy

            # NumPy/OpenBLAS and jupyter-client -> zmq runtime libraries.
            stdenv.cc.cc.lib
            zlib

            # playwright
            chromium
            nodejs_22
            code-server

            # desktop gui runtime
            novnc
            scrot
            xdotool
            x11vnc
            xterm
            xorg.xev
            xorg.xhost
            xorg.xmessage
            xorg.xrandr
            xorg.xsetroot
            xorg.xvfb

            # linux utils
            coreutils
            xfsprogs
            util-linux
            findutils
            gawk
            gnused
            gnutar
            gzip
            git
            curl
            which
            tmux
            ripgrep
            gnupatch
            zsh
            jq
          ]) ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isAarch64 [
            # uvloop-snapshot has no Linux ARM64 wheels and builds bundled libuv.
            pkgs.stdenv.cc
            pkgs.stdenv.cc.bintools
            pkgs.gnumake
          ];
          postBuild = ''
            # 创建 fontconfig 配置
            mkdir -p $out/etc/fonts
            cp ${fontsConf} $out/etc/fonts/fonts.conf

            # agent-bin：面向 agent 暴露的工具白名单目录。
            # get_clean_sandbox_env() 会剥掉整个 nix bin（让 agent 看用户镜像原生
            # 工具链），但仍把这个目录追加到 PATH 末尾，使 docker 等平台提供的工具
            # 无需 PATH 前缀即可用，且用户镜像同名工具仍优先。用相对软链以便随
            # $out 重定位到 /nix/runtime/nix-env 后仍有效。新增要暴露的工具加进下表即可。
            mkdir -p $out/agent-bin
            for t in docker dockerd docker-buildx docker-compose ctr runc; do
              if [ -e "$out/bin/$t" ]; then
                ln -s "../bin/$t" "$out/agent-bin/$t"
              fi
            done
          '';
          ignoreCollisions = true;
        };
    });
  };
}
