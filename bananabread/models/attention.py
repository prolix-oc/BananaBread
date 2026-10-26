"""Hardware eligibility for the supported FlashAttention 2 builds."""

import torch

# GPU targets bundled by mjun0812's commit-pinned ROCm CK wheels.
ROCM_FLASH_ARCHS = {
    "gfx90a",
    "gfx942",
    "gfx950",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1150",
    "gfx1151",
    "gfx1200",
    "gfx1201",
}


def flash_attention_supported_device(device: str | torch.device) -> bool:
    if isinstance(device, str) and device == "auto":
        device = "cuda"
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None):
        arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":")[0]
        return arch in ROCM_FLASH_ARCHS
    return torch.cuda.get_device_capability(device)[0] >= 8
