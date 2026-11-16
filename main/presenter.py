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

import numpy as np


class RealtimePresenter:
    """每一帧模拟器画面都画出来，并用这一帧的音频卡住播放速度。"""

    def __init__(self, sample_rate, frame_rate):
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
        # vsync 关掉：刷新率不一定是 59.92，用音频时钟来限速更接近原机。
        self.window = pyglet.window.Window(
            fullscreen=True,
            screen=screen,
            vsync=False,
            caption="Street Fighter II",
        )
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

    def present(self, frame, audio):
        """画一帧，再把这一帧对应的音频送出去。窗口被关掉时结束整个进程。"""
        if self.closed:
            raise SystemExit(0)
        self._draw(frame)
        self._play(audio)

    def close(self):
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
