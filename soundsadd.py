from gtts import gTTS
import os

os.makedirs("sounds", exist_ok=True)

for ch in range(ord('A'), ord('Z') + 1):
    letter = chr(ch)
    tts = gTTS(text=f"Letter {letter}", lang='en')
    tts.save(f"sounds/{letter}.mp3")

print("✅ All sounds created in /sounds folder!")
