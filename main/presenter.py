# Copyright 2026 Lin Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""把模拟器画面和声音按游戏原速播出来。

训练和快速预览仍走 OpenCV 小窗。只有测试加上 --realtime 时，
包装器才会用这里的全屏窗口：画面按比例放到最大，声音跟画面同步。
"""

import time
from pathlib import Path

import numpy as np


def _ffmpeg_tool():
    """系统里没有 ffmpeg 时，用这台机器上能找到动态库的那一份。"""
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found, None
    studio = Path("/Applications/Studio 2.0/PhotoRealisticRenderer/mac/ffmpeg")
    if studio.is_file():
        return str(studio), str(studio.parent)
    return None, None


class RealtimePresenter:
    """每一帧模拟器画面都画出来，并用这一帧的音频卡住播放速度。"""

    def __init__(self, sample_rate, frame_rate, record_path=None):
        import pyglet
        from pyglet.gl import gl
        import sounddevice as sd

        # pyglet 自己也会打开音频设备。这里只拿它画图，声音交给 sounddevice，
        # 避免两个库抢同一块声卡。
        pyglet.options["audio"] = ("silent",)
        self._pyglet = pyglet
        self._gl = gl
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        # 不用独占全屏。独占全屏会把 Command-Tab 这类系统按键一起吃掉，切不走。
        # 无边框铺满屏幕，看起来仍是一整块画面，系统快捷键留给系统。
        self.window = pyglet.window.Window(
            width=screen.width,
            height=screen.height,
            screen=screen,
            style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS,
            fullscreen=False,
            vsync=False,
            caption="Street Fighter II",
        )
        self.window.set_location(screen.x, screen.y)
        if hasattr(self.window, "set_exclusive_keyboard"):
            self.window.set_exclusive_keyboard(False)
        self.window.activate()
        self.closed = False

        @self.window.event
        def on_close():
            self.closed = True

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.ESCAPE:
                self.closed = True

        # 世嘉 MD 的 NTSC 帧率，模拟器读不到时用这个兜底。
        self.frame_rate = float(frame_rate) if frame_rate else 59.92274340431231
        self._next_frame = time.perf_counter()
        # 纹理只建一次，后面每帧只更新像素，避免长时间播放把显存撑满。
        self._texture = None
        self._audio = None
        self._record_path = None
        self._record_video_path = None
        self._record_wav_path = None
        self._record_video = None
        self._record_wav = None
        self._record_rate = int(round(sample_rate)) if sample_rate else 44100
        if record_path:
            from pathlib import Path
            self._record_path = Path(record_path)
            self._record_video_path = Path("/tmp/sf2_arcade_picture.mp4")
            self._record_wav_path = Path("/tmp/sf2_arcade_sound.wav")
        try:
            # 大约 100 毫秒缓冲，用来盖过模型推理那几毫秒的抖动。
            # write() 会在缓冲区满了以后堵住，播放速度就被声卡锁在原速。
            self._audio = sd.OutputStream(
                samplerate=int(round(sample_rate)),
                channels=2,
                dtype="int16",
                latency=0.1,
            )
            self._audio.start()
        except Exception as exc:
            self._audio = None
            print(f"Audio unavailable, pacing video only: {exc}", flush=True)
        print(
            f"Playback window {self.window.width}x{self.window.height}, "
            f"{self.frame_rate:.2f} fps. Press Esc to quit.",
            flush=True,
        )
        if self._record_path is not None:
            print(f"Recording to {self._record_path}", flush=True)

    def present(self, frame, audio):
        """画一帧，再把这一帧对应的音频送出去。窗口被关掉时结束整个进程。"""
        if self.closed:
            raise SystemExit(0)
        self._draw(frame)
        self._record(frame, audio)
        self._play(audio)

    def close(self):
        self._finish_record()
        if self._audio is not None:
            try:
                self._audio.stop()
                self._audio.close()
            except Exception:
                pass
            self._audio = None
        window = getattr(self, "window", None)
        if window is not None:
            try:
                window.close()
            except Exception:
                pass
            self.window = None

    def _draw(self, frame):
        """按屏幕能放下的最大尺寸居中画，左右或上下留黑边，不把画面拉变形。"""
        gl = self._gl
        window = self.window
        height, width = frame.shape[:2]
        # pitch 为负：OpenGL 的原点在左下，游戏画面原点在左上，这样翻一下才是正的。
        image = self._pyglet.image.ImageData(
            width,
            height,
            "RGB",
            np.ascontiguousarray(frame).tobytes(),
            pitch=width * -3,
        )
        window.switch_to()
        window.dispatch_events()
        if self.closed:
            raise SystemExit(0)
        if self._texture is None:
            self._texture = image.get_texture()
            from pyglet.gl import GL_FALSE, GL_GENERATE_MIPMAP
            gl.glBindTexture(self._texture.target, self._texture.id)
            # 关掉多级纹理，放大时取最近的像素。全屏拉大仍是一块块像素，不会发糊。
            gl.glTexParameteri(gl.GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_FALSE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        else:
            gl.glBindTexture(self._texture.target, self._texture.id)
            image.blit_to_texture(self._texture.target, self._texture.level, 0, 0, 0)
        scale = min(window.width / width, window.height / height)
        draw_w = max(1, int(width * scale))
        draw_h = max(1, int(height * scale))
        x = (window.width - draw_w) // 2
        y = (window.height - draw_h) // 2
        window.clear()
        self._texture.blit(x, y, width=draw_w, height=draw_h)
        window.flip()

    def _play(self, audio):
        """有声卡时用音频堵住调用方；没有声卡时按帧率 sleep，速度仍然是原速。"""
        samples = None
        if audio is not None and len(audio):
            # 复制一份。模拟器下一次 step 会覆盖这块内存。
            samples = np.array(audio, dtype=np.int16, copy=True)
            if samples.ndim == 1:
                samples = samples.reshape(-1, 2)
        if self._audio is not None and samples is not None:
            self._audio.write(samples)
            return
        self._next_frame += 1.0 / self.frame_rate
        delay = self._next_frame - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            # 这一帧已经迟到了，把时钟拉回现在，避免后面连续快进补帧。
            self._next_frame = time.perf_counter()

    def _record(self, frame, audio):
        """把正在播放的这一帧和它的声音写下来。窗口仍是铺满屏幕的那一块。

        先按整数倍最近邻放大，再写文件。原生小图直接做色度抽样时，
        相邻两颗颜色不同的像素会被合成一颗，红和蓝就会偏。
        """
        if self._record_path is None:
            return
        import cv2
        import wave
        picture = np.ascontiguousarray(frame[:, :, :3])
        height, width = picture.shape[:2]
        scale = 4 if max(width, height) <= 400 else 1
        if scale > 1:
            picture = cv2.resize(
                picture,
                (width * scale, height * scale),
                interpolation=cv2.INTER_NEAREST,
            )
        if self._record_video is None:
            height, width = picture.shape[:2]
            writer = cv2.VideoWriter(
                str(self._record_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.frame_rate,
                (width, height),
            )
            if not writer.isOpened():
                print(f"Recording failed to open {self._record_video_path}", flush=True)
                self._record_path = None
                return
            self._record_video = writer
            wav = wave.open(str(self._record_wav_path), "wb")
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(self._record_rate)
            self._record_wav = wav
        bgr = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)
        self._record_video.write(bgr)
        if audio is None or len(audio) == 0 or self._record_wav is None:
            return
        samples = np.array(audio, dtype=np.int16, copy=True)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 2)
        self._record_wav.writeframes(samples.tobytes())

    def _finish_record(self):
        """把画面和声音合成一条有声的 mp4。窗口关掉时也要收完，否则文件打不开。"""
        video = self._record_video
        wav = self._record_wav
        self._record_video = None
        self._record_wav = None
        if video is not None:
            video.release()
        if wav is not None:
            wav.close()
        final = self._record_path
        picture = self._record_video_path
        sound = self._record_wav_path
        if final is None or picture is None or not picture.is_file():
            return
        import shutil
        import subprocess
        ffmpeg, ffmpeg_cwd = _ffmpeg_tool()
        # 画面流原样拷进成片，不再做一次色度抽样。声音另压成 aac。
        if ffmpeg and sound is not None and sound.is_file() and sound.stat().st_size > 44:
            final.parent.mkdir(parents=True, exist_ok=True)
            done = subprocess.run(
                [
                    ffmpeg, "-y",
                    "-i", str(picture),
                    "-i", str(sound),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    str(final),
                ],
                cwd=ffmpeg_cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if done.returncode == 0 and final.is_file():
                picture.unlink(missing_ok=True)
                sound.unlink(missing_ok=True)
                print(f"Saved {final}", flush=True)
                return
        if picture.is_file():
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(picture), str(final))
            print(f"Saved picture without sound: {final}", flush=True)
