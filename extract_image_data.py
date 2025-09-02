import cv2
import numpy as np
from itertools import groupby

def kmeans_1d(values, k=3, iters=50):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.array([]), np.array([], dtype=int)
    centers = np.percentile(arr, np.linspace(0, 100, k))
    for _ in range(iters):
        d = np.abs(arr.reshape(-1, 1) - centers.reshape(1, -1))
        labels = np.argmin(d, axis=1)
        new_centers = np.array([arr[labels == i].mean() if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers, labels

def extract_by_projection(bw, debug=False, img_color=None):
    h, w = bw.shape
    rows = np.where(bw.sum(axis=1) > 0)[0]
    if rows.size == 0:
        return ""
    top, bottom = rows[0], rows[-1]
    band = bw[max(0, top-3):min(h, bottom+3), :]
    proj = (band.sum(axis=0) > 0).astype(np.uint8)

    runs = [(k, sum(1 for _ in g)) for k, g in groupby(proj)]
    mark_lengths = [length for k, length in runs if k == 1]
    gap_lengths = [length for k, length in runs if k == 0]

    # classify marks -> dot/dash (k=2)
    if len(mark_lengths) >= 2:
        centers_m, labels_m = kmeans_1d(mark_lengths, k=2)
        order_m = np.argsort(centers_m)
        inv_m = np.empty_like(order_m)
        for i, orig in enumerate(order_m):
            inv_m[orig] = i
        mapped_marks = [int(inv_m[l]) for l in labels_m]
    else:
        med = np.median(mark_lengths) if mark_lengths else 1
        mapped_marks = [0 if l <= med * 1.5 else 1 for l in mark_lengths]

    # cluster gaps -> intra / between-letter / between-word (k=3)
    if len(gap_lengths) >= 3:
        centers_g, labels_g = kmeans_1d(gap_lengths, k=3)
        order_g = np.argsort(centers_g)
        inv_g = np.empty_like(order_g)
        for i, orig in enumerate(order_g):
            inv_g[orig] = i
        mapped_gaps = [int(inv_g[l]) for l in labels_g]
    else:
        medg = np.median(gap_lengths) if gap_lengths else 1.0
        mapped_gaps = []
        for g in gap_lengths:
            if g <= medg * 0.6:
                mapped_gaps.append(0)
            elif g <= medg * 1.8:
                mapped_gaps.append(1)
            else:
                mapped_gaps.append(2)

    morse = ""
    mark_idx = 0
    gap_idx = 0
    seq = runs
    for k, length in seq:
        if k == 1:
            lab = mapped_marks[mark_idx] if mark_idx < len(mapped_marks) else 0
            morse += '.' if lab == 0 else '-'
            mark_idx += 1
        else:
            lab = mapped_gaps[gap_idx] if gap_idx < len(mapped_gaps) else 0
            if lab == 1:
                morse += " "
            elif lab == 2:
                morse += "  "
            gap_idx += 1
    return morse
'''
PLOTTING THE MORSE CODE IMAGE TO DEBUG
    if debug and img_color is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,1, figsize=(10,3), gridspec_kw={'height_ratios':[3,1]})
        ax[0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)); ax[0].axis('off')
        ax[1].plot(proj, drawstyle='steps-mid'); ax[1].set_ylim(-0.1,1.1); ax[1].set_yticks([])
        xpos = 0
        m_idx=0
        for k, length in runs:
            if k == 1:
                lab = mapped_marks[m_idx] if m_idx < len(mapped_marks) else 0
                ax[1].text(xpos + length/2, 0.5, '.' if lab==0 else '-', ha='center', va='center')
                m_idx += 1
            xpos += length
        plt.tight_layout(); plt.show()
'''
    
def extract_by_contours(gray, debug=False, img_color=None):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 9)
    h, w = th.shape
    kern_w = max(3, w // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_w, 3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x, y, rw, rh = cv2.boundingRect(c)
        if rw * rh < 30:
            continue
        rects.append((x, y, rw, rh))
    rects = sorted(rects, key=lambda r: r[0])

    if rects:
        median_h = float(np.median([r[3] for r in rects]))
    else:
        median_h = 1.0
    merge_gap = max(2, int(median_h * 0.6))
    merged = []
    for (x, y, rw, rh) in rects:
        if not merged:
            merged.append([x, y, rw, rh])
            continue
        px, py, pw, ph = merged[-1]
        gap = x - (px + pw)
        top = max(py, y); bottom = min(py+ph, y+rh)
        overlap = max(0, bottom - top)
        overlap_ratio = overlap / min(ph, rh) if min(ph, rh) > 0 else 0
        if gap <= merge_gap and overlap_ratio > 0.1:
            nx = min(px, x); ny = min(py, y)
            nw = max(px+pw, x+rw) - nx
            nh = max(py+ph, y+rh) - ny
            merged[-1] = [nx, ny, nw, nh]
        else:
            merged.append([x, y, rw, rh])

    boxes = merged
    boxes = sorted(boxes, key=lambda r: r[0])
    widths = np.array([b[2] for b in boxes]) if boxes else np.array([1.0])
    median_w = float(np.median(widths))
    symbol_thresh = max(1.0, median_w * 1.4)

    symbols = []
    for (x, y, rw, rh) in boxes:
        sym = "." if rw < symbol_thresh else "-"
        symbols.append({'x': x, 'y': y, 'w': rw, 'h': rh, 'sym': sym})

    gaps = [max(0, symbols[i+1]['x'] - (symbols[i]['x'] + symbols[i]['w'])) for i in range(len(symbols)-1)]
    if len(gaps) >= 3:
        centers, labels = kmeans_1d(gaps, k=3)
        order = np.argsort(centers); inv = np.empty_like(order)
        for i, orig in enumerate(order):
            inv[orig] = i
        mapped = [int(inv[l]) for l in labels]
    else:
        medg = np.median(gaps) if gaps else 1.0
        mapped = []
        for g in gaps:
            if g <= medg * 0.6:
                mapped.append(0)
            elif g <= medg * 1.8:
                mapped.append(1)
            else:
                mapped.append(2)

    morse = ""
    for i, s in enumerate(symbols):
        morse += s['sym']
        if i < len(gaps):
            lab = mapped[i]
            if lab == 1:
                morse += " "
            elif lab == 2:
                morse += "  "

    # if debug and img_color is not None:
    #     import matplotlib.pyplot as plt
    #     v = img_color.copy()
    #     for s in symbols:
    #         x, y, w2, h2 = int(s['x']), int(s['y']), int(s['w']), int(s['h'])
    #         cv2.rectangle(v, (x, y), (x+w2, y+h2), (0,255,0) if s['sym']=='.' else (0,0,255), 2)
    #         cv2.putText(v, s['sym'], (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    #     # plt.figure(figsize=(10,2)); plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()

    return morse

def extract_morse_auto(image_path, debug=True):
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    rows = np.where(bw.sum(axis=1) > 0)[0]
    if rows.size == 0:
        return ""
    content_height = rows[-1] - rows[0] + 1
    H = bw.shape[0]

    if content_height <= max(8, H * 0.22):
        if debug: print("Auto: using projection method (typed-looking image)")
        return enhance_morse_code(extract_by_projection(bw, debug=debug, img_color=img_color))
    else:
        if debug: print("Auto: using contour/robust method (handwritten-looking image)")
        return enhance_morse_code(extract_by_contours(gray, debug=debug, img_color=img_color))

def enhance_morse_code(result):
    i = 0
    while i < (len(result) - 1):
        if result[i:i+2] == '  ' and result[i+2] != ' ':
            result = result[0:i+2] + ' ' + result[i+2:]
            i = i+2
        i+=1
    return result

