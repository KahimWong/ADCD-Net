from PIL import Image
import numpy as np

def get_luma_qt_8x8(img_or_path):
    """
    Return the JPEG luma (Y) quantization table as an (8, 8) numpy array.
    If not JPEG or no quantization tables, return None.

    Notes:
      - Pillow exposes JPEG quantization tables via `im.quantization`.
      - The table values are typically in JPEG zigzag order; this converts to 8x8 natural order.
    """
    # Zigzag -> row-major index map (length 64)
    zz = []
    n = 8
    for s in range(2 * n - 1):
        if s % 2 == 0:
            r = min(s, n - 1)
            c = s - r
            while r >= 0 and c < n:
                zz.append(r * n + c)
                r -= 1
                c += 1
        else:
            c = min(s, n - 1)
            r = s - c
            while c >= 0 and r < n:
                zz.append(r * n + c)
                r += 1
                c -= 1

    def _to_rowmajor_8x8(q64_zigzag):
        # q64_zigzag[i] corresponds to row-major position zz[i]
        out = np.empty(64, dtype=np.int32)
        for i, rm_idx in enumerate(zz):
            out[rm_idx] = int(q64_zigzag[i])
        return out.reshape(8, 8)

    # Open image if a path is provided
    close_after = False
    if isinstance(img_or_path, Image.Image):
        im = img_or_path
    else:
        im = Image.open(img_or_path)
        close_after = True

    try:
        if im.format != "JPEG":
            return None

        qtables = getattr(im, "quantization", None)
        if not qtables:
            return None

        # Luma table is typically id 0; fallback to smallest key if missing.
        tid = 0 if 0 in qtables else min(qtables.keys())
        q64 = qtables.get(tid)
        if q64 is None or len(q64) != 64:
            return None

        return _to_rowmajor_8x8(q64)
    finally:
        if close_after:
            im.close()
