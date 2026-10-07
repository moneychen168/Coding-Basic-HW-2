import asyncio, json, os, sys, edge_tts
d = json.load(open("slides.json", encoding="utf-8"))
os.makedirs("audio", exist_ok=True)
only = int(sys.argv[1]) if len(sys.argv) > 1 else None   # 1-based, 只做單張
async def main():
    for i, s in enumerate(d["slides"], 1):
        if only and i != only: continue
        await edge_tts.Communicate(s["narr"], d.get("voice", "zh-TW-HsiaoChenNeural")).save(f"audio/s{i:02d}.mp3")
        print("ok", i, flush=True)
asyncio.run(main())
