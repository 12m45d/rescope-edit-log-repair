#!/bin/bash
# usage: then_chain.sh <pid of a running worker of mine> <chain_resume.sh arguments ...>
# CPU only: waits until that worker has ended (it is never signalled), then becomes chain_resume.sh with the given
# arguments.  Used on 2026-11-10 19:35 to change the ORDER of the remaining units of a running resume chain to the
# order of the task (every D2 decode before the text head-to-head) without interrupting the worker and without a
# second process of this experiment on the card: the old chain shell is stopped, its worker finishes, this follows.
cd "$(dirname "$0")"
pid=$1; shift
while kill -0 "$pid" 2>/dev/null; do sleep 5; done
exec ./chain_resume.sh "$@"
