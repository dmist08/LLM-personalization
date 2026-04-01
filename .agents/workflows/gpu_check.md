---
description: Check GPU status before launching any training job
---
// turbo-all
1. Run: `nvidia-smi`
2. Report: GPU model, total VRAM, used VRAM, running processes
3. If used VRAM > 20GB: warn me and do NOT launch any training job until I confirm
