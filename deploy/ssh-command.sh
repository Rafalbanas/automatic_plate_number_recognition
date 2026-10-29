#!/usr/bin/env bash
#
# Wrapper for the ssh-command unit. SSH_ORIGINAL_COMMAND is validated with
# fixed regular expressions and forwarded only as exact, pre-tokenised
# arguments to the root-owned receiver at /usr/local/sbin/plates-deploy.
# Anything else exits 2; nothing is ever eval'ed.

set -euo pipefail
original="${SSH_ORIGINAL_COMMAND:-}"

if [[ $original == status ]]; then
    exec sudo -n /usr/local/sbin/plates-deploy status
fi
if [[ $original =~ ^deploy\ ([a-f0-9]{40})\ ([a-f0-9]{64})$ ]]; then
    exec sudo -n /usr/local/sbin/plates-deploy "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
fi
exit 2
