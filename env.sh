# Per-shell environment for new_gen_R.  Usage:  source env.sh
#
# Sources the haxad_demonstrator's own setup.sh (conda env, PYTHONPATH incl.
# submodules, HAXAD_STORE, law config), so the demonstrator's tasks / data
# helpers are importable and resolve the same store paths as the pipeline.
# The demonstrator code is imported, never modified.
#
# PYTHONNOUSERSITE is load-bearing: ~/.local ships numpy 2.x, which breaks the
# env's numexpr/pytables (numpy 1.26 ABI). Set before setup.sh, whose torch
# probe already runs python.

_ngr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONNOUSERSITE=1
export HAXAD_CODE_DIR="${HAXAD_CODE_DIR:-/pscratch/sd/l/liangyu/HAXAD/haxad_demonstrator}"

type module >/dev/null 2>&1 && module load python/3.10 >/dev/null 2>&1
source "${HAXAD_CODE_DIR}/setup.sh" >/dev/null

export NGR_DIR="${_ngr_dir}"
export PYTHONPATH="${_ngr_dir}:${PYTHONPATH}"

# Where caches / runs go.
export NGR_STORE="${NGR_STORE:-${_ngr_dir}/store}"
mkdir -p "${NGR_STORE}"

unset _ngr_dir
