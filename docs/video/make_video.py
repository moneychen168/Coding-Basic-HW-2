"""從 slides.json + audio/sNN.mp3 合成 report_1007.mp4 (1920x1080)。
slides.json schema:
 {"voice": str, "slides": [ {"title": str,
     "bullets": [str,...]   # 最多 5 條 (與 table 擇一)
     "table": [[hdr...],[row...],...]  # 第一列為表頭
     "foot": str (選填, 底部註解),
     "narr": str (旁白; tts.py 用) } ]}
無 audio/sNN.mp3 時該張靜音停 8 秒。"""
import json, os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
os.chdir(os.path.dirname(os.path.abspath(__file__)))
W, H = 1920, 1080
FONT = "fonts/NotoSansTC.ttf"
def font(sz, w="Bold"):
    f = ImageFont.truetype(FONT, sz)
    try: f.set_variation_by_name(w)
    except Exception: pass
    return f
def wrap(d, text, f, maxw):
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=f) > maxw and cur:
            lines.append(cur); cur = ch
        else: cur += ch
    return lines + [cur]
def draw(s, idx, n, path):
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 150], fill=(20, 33, 61))
    d.text((70, 75), s["title"], font=font(64), fill="white", anchor="lm")
    d.text((W-50, H-30), f"{idx}/{n}", font=font(30, "Regular"), fill=(120, 120, 120), anchor="rb")
    if "table" in s:
        t = s["table"]; cw = [640, 400, 460, 280]; rh = 110; x0, y = 70, 210
        for r, row in enumerate(t):
            x = x0
            for c, cell in enumerate(row):
                if r == 0: d.rectangle([x, y, x+cw[c], y+rh], fill=(20, 33, 61))
                elif r % 2 == 0: d.rectangle([x, y, x+cw[c], y+rh], fill=(235, 240, 248))
                d.text((x+20, y+rh//2), cell, font=font(40 if r else 38), anchor="lm",
                       fill="white" if r == 0 else (20, 33, 61))
                x += cw[c]
            y += rh
        y += 30
    else:
        y = 220; f = font(46, "Medium")
        for b in s["bullets"][:5]:
            ls = wrap(d, b, f, W-220)
            d.ellipse([80, y+18, 100, y+38], fill=(214, 90, 40))
            for l in ls:
                d.text((125, y), l, font=f, fill=(25, 25, 25)); y += 68
            y += 38
    if s.get("foot"):
        d.text((70, H-80), s["foot"], font=font(38, "Regular"), fill=(80, 80, 80), anchor="lm")
    im.save(path)
def main():
    sl = json.load(open("slides.json", encoding="utf-8"))["slides"]
    ff = imageio_ffmpeg.get_ffmpeg_exe(); os.makedirs("build", exist_ok=True)
    segs = []
    for i, s in enumerate(sl, 1):
        png, mp3, seg = f"build/s{i:02d}.png", f"audio/s{i:02d}.mp3", f"build/s{i:02d}.mp4"
        draw(s, i, len(sl), png)
        base = [ff, "-y", "-loop", "1", "-framerate", "10", "-i", png]
        if os.path.exists(mp3):
            cmd = base + ["-i", mp3, "-af", "apad=pad_dur=0.8", "-shortest"]
        else:
            cmd = base + ["-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "8"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "10", "-c:a", "aac", "-ar", "24000", "-ac", "1", seg]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL); segs.append(seg)
    open("build/list.txt", "w").write("".join(f"file '{os.path.abspath(p)}'\n" for p in segs))
    subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", "build/list.txt", "-c", "copy", "report_1007.mp4"], check=True, stderr=subprocess.DEVNULL)
    print("done report_1007.mp4")
if __name__ == "__main__": main()
