from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from .game import Config, DOWN, STAY, UP
from .gif import GifRecorder
from .highlights import clip_frames, scout
from .prediction import predicted_cells


class HighlightTests(unittest.TestCase):
    def test_rackets_hide_predictions_and_earliest_token_wins(self):
        config = Config()
        queues = [[DOWN]*6+[UP]*4, [STAY]*10]
        predictions = predicted_cells([11.5, 11.5], queues, [0, 0], config)
        cells = {(p['row'], p['col']): p['number'] for p in predictions}
        self.assertEqual(len(cells), len(predictions))
        self.assertEqual(cells[(16, 1)], 5)  # Steps 6 and 7 revisit this cell.
        self.assertTrue(all(p['side'] == 0 for p in predictions))  # Right STAY is covered by racket.
        self.assertTrue(all(p['row'] not in range(10, 15) for p in predictions))
        without_ball = predicted_cells([11.5, 11.5], queues, [0, 0], config, ball=(16, 1))
        self.assertFalse(any((p['row'], p['col']) == (16, 1) for p in without_ball))
        self.assertEqual(predicted_cells([3, 20], [[UP]*10, [DOWN]*10], [0, 0], config), [])

    def test_scout_preserves_terminal_frame_and_queues_across_points(self):
        class StillModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.calls = 0

            def generate(self, observations):
                self.calls += 1
                moves = torch.zeros(len(observations), 10, dtype=torch.long)
                return moves, torch.nn.functional.one_hot(moves, 3).float()

        model = StillModel()
        with redirect_stdout(io.StringIO()):
            match = scout(model, Config(), games=1, ticks=250, seed=42)[0]
        self.assertEqual(model.calls, 25)
        self.assertEqual(match['refills'], [25, 25])
        endings = [s for s in match['states'] if s.scorer is not None]
        self.assertGreater(len(endings), 0)
        for state in endings:
            self.assertTrue(state.x < 0 or state.x > 31)
            self.assertEqual(state.positions, ((state.tick-1)%10+1,)*2)
        for rally in match['rallies']:
            clip = dict(start=rally['start'], end=rally['end'], event=rally['end'])
            frames = list(clip_frames(clip))
            self.assertEqual(frames[-1][0], rally['end'])
            self.assertEqual(len({match['states'][i].rally for i, _ in frames}), 1)

    def test_streamed_gif_keeps_black_frames_and_hold_durations(self):
        palette_source = Image.new('RGB', (8, 4), 'black')
        palette_source.paste((255, 207, 94), (4, 0, 8, 4))
        palette = palette_source.quantize(colors=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.gif'
            writer = GifRecorder(path, 10, palette)
            writer.append_image(Image.new('RGB', (8, 4), 'black'), 3)
            writer.append_image(Image.new('RGB', (8, 4), (255, 207, 94)), 2)
            writer.close()
            with Image.open(path) as image:
                self.assertEqual(image.info['loop'], 0)
                self.assertEqual(image.convert('RGB').getpixel((0, 0)), (0, 0, 0))
                self.assertEqual(image.info['duration'], 300)
                image.seek(1)
                self.assertEqual(image.convert('RGB').getpixel((0, 0)), (255, 207, 94))
                self.assertEqual(image.info['duration'], 200)


if __name__ == '__main__':
    unittest.main()
