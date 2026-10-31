"""Stream a shared-palette, cropped-frame GIF without buffering the movie."""
from pathlib import Path

from PIL import GifImagePlugin, Image, ImageChops


class GifRecorder:
    def __init__(self, path, fps, palette=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_suffix('.gif.tmp')
        self.stream = self.temporary.open('wb')
        self.fps = fps
        self.frames = 0
        self.palette, self.previous = palette, None

    def append(self, surface):
        import pygame
        rgb = Image.frombytes('RGB', surface.get_size(), pygame.image.tobytes(surface, 'RGB'))
        self.append_image(rgb)

    def append_image(self, rgb, repeat=1):
        if self.previous is None:
            if self.palette is None:
                frame = rgb.quantize(colors=128, dither=Image.Dither.NONE)
                self.palette = frame.copy()
            else:
                frame = rgb.quantize(palette=self.palette, dither=Image.Dither.NONE)
            header, _ = GifImagePlugin.getheader(frame, info={'loop': 0})
            for block in header:
                self.stream.write(block)
            box = (0, 0, *frame.size)
        else:
            frame = rgb.quantize(palette=self.palette, dither=Image.Dither.NONE)
            box = ImageChops.difference(frame.convert('RGB'), self.previous.convert('RGB')).getbbox()
            box = box or (0, 0, 1, 1)
        # GIF delays are whole centiseconds: distribute rounding for exact timing.
        delay = (round((self.frames + repeat) * 100 / self.fps) -
                 round(self.frames * 100 / self.fps)) * 10
        for block in GifImagePlugin.getdata(frame.crop(box), offset=box[:2], duration=delay, disposal=1):
            self.stream.write(block)
        self.previous = frame
        self.frames += repeat

    def close(self):
        self.stream.write(b';')
        self.stream.close()
        self.temporary.replace(self.path)
