import json
import os
import sys
from pathlib import Path
import time

import torch

if __name__ == "__main__":
    stats_path = Path(sys.argv.pop(1))
    import boltz.main as bm

    scale = float(os.environ.get("NOISE_SCALE", "0"))
    _Orig = bm.Boltz2DiffusionParams

    def _params():
        params = _Orig()
        params.noise_scale = scale
        return params

    bm.Boltz2DiffusionParams = _params
    print("PATCHED noise_scale=", bm.Boltz2DiffusionParams().noise_scale,
          file=sys.stderr, flush=True)

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        from boltz.main import cli
        cli()
    finally:
        stats_path.write_text(json.dumps({
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "elapsed_seconds_after_torch_import": time.time() - started,
        }, indent=2))
