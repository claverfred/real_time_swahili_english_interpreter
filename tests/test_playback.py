import asyncio
import time

import numpy as np

from src.pipeline.events import EventRecorder, EventType
from src.pipeline.playback import AudioPlayer


def test_player_plays_in_order_and_counts_underruns():
    async def go():
        rec = EventRecorder()
        played = []

        def fake_play(audio, sr):
            played.append(len(audio))
            time.sleep(len(audio) / sr)

        player = AudioPlayer(rec, pause_ms=0, play_fn=fake_play, underrun_gap_ms=50)
        player.start()
        tone = np.ones(1000, np.float32) * 0.1           # 0.1 s at 10 kHz
        player.offer(rec.emit(EventType.TTS_AUDIO, audio=tone, unit_id=0, chunk=0, sample_rate=10000))
        await asyncio.sleep(0.3)                          # chunk 1 arrives 0.2 s after chunk 0 ended
        player.offer(rec.emit(EventType.TTS_AUDIO, audio=tone, unit_id=0, chunk=1, sample_rate=10000))
        player.offer(rec.emit(EventType.TTS_AUDIO, audio=tone, unit_id=1, chunk=0, sample_rate=10000))
        await player.close()
        s = player.stats()
        assert played == [1000, 1000, 1000] and s["chunks"] == 3
        assert s["underruns"] == 1 and s["underrun_gap_ms"]["min"] > 100   # new unit is not an underrun
        assert s["playback_delay_ms"]["max"] >= 50        # unit 1 waited behind unit 0's chunk
        assert abs(s["played_s"] - 0.3) < 1e-6 and set(player.first_played_ms) == {0, 1}
    asyncio.run(go())


def test_is_speaking_covers_playback_and_echo_tail():
    async def go():
        rec = EventRecorder()
        player = AudioPlayer(rec, pause_ms=0, play_fn=lambda a, sr: time.sleep(0.2))
        assert not player.is_speaking()
        player.start()
        player.offer(rec.emit(EventType.TTS_AUDIO, audio=np.zeros(10, np.float32), unit_id=0, chunk=0))
        assert player.is_speaking()                       # queued
        await asyncio.sleep(0.1)
        assert player.is_speaking()                       # playing
        await asyncio.sleep(0.2)
        assert player.is_speaking(tail_ms=400)            # just ended: room echo tail
        assert not player.is_speaking(tail_ms=0)
        await player.close()
    asyncio.run(go())


def test_player_cancel_drops_queue():
    async def go():
        rec = EventRecorder()
        player = AudioPlayer(rec, play_fn=lambda a, sr: time.sleep(0.05), stop_fn=lambda: None)
        for i in range(5):
            player.offer(rec.emit(EventType.TTS_AUDIO, audio=np.zeros(10, np.float32), unit_id=i, chunk=0))
        player.cancel()
        player.start()
        await player.close()
        assert player.stats()["chunks"] == 0
    asyncio.run(go())
