"""Record agent Bash commands for OSWorld's command-history getters.

Only source this in Codex's child shells, never the installer/launcher/poller.
Do not inject expected task commands or truncate existing shell history.
"""

# DEBUG records Bash's command representation, not a terminal
# transcript. Python-only edits, zsh/sh and explicitly disabled hooks are not
# covered. A failed history write must not replace the command's exit status.
BASH_HISTORY_HOOK = r"""if [ -n "${HOME:-}" ]; then
    trap 'case "$BASH_COMMAND" in *bash_history*) ;; *) printf "%s\n" "$BASH_COMMAND" 2>/dev/null >> "$HOME/.bash_history" || : ;; esac' DEBUG
fi
"""
