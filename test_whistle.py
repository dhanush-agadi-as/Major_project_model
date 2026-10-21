import wave, needle, speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone(sample_rate=16000) as src:
    r.adjust_for_ambient_noise(src, duration=1)
    print("Say: hey vision remember this phone")
    audio = r.listen(src, timeout=8, phrase_time_limit=5)

with wave.open("test.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(audio.get_raw_data(convert_rate=16000, convert_width=2))
import stt_engine
keywords = stt_engine.get_keywords()
print(needle.transcribe("test.wav", language="en", keywords=keywords))