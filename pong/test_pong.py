import tempfile
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
import unittest

import numpy as np
import torch

from .evaluate import match
from .game import Config, DOWN, Pong, STAY, UP, expert, tracker
from .model import BOS, ModelPlayer, PongTransformer, load
from .prepare_data import future_actions, pack, record
from .train import decoder_inputs, unpack


class PongTests(unittest.TestCase):
    def test_numeric_history_and_mirroring(self):
        game = Pong(12)
        original = game.observation()
        self.assertEqual(original.shape, (4, 24, 32))
        self.assertEqual(original.dtype, np.uint8)
        self.assertEqual(set(np.unique(original)), {0, 1, 2, 3})
        game.step(UP, DOWN)
        np.testing.assert_array_equal(game.observation()[:-1], original[1:])
        np.testing.assert_array_equal(game.observation(1), game.observation()[:, :, ::-1])
        original[:] = 0
        self.assertNotEqual(game.observation().sum(), 0)

    def test_walls_paddles_and_scoring(self):
        game = Pong()
        game.x, game.y, game.vx, game.vy = 10, 1.1, -0.7, -0.5
        game.step(STAY, STAY)
        self.assertAlmostEqual(game.y, 1.4)
        self.assertGreater(game.vy, 0)
        game.x, game.y, game.vx, game.vy = 2.3, game.paddles[0], -0.7, 0
        game.step(STAY, STAY)
        self.assertGreater(game.vx, 0)
        self.assertEqual(game.hits, [1, 0])
        game.x, game.y, game.vx = 0.1, 2, -0.7
        self.assertEqual(game.step(STAY, STAY), 1)
        self.assertEqual(game.score, [0, 1])
        self.assertEqual(len(game.frames), 4)
        for _ in range(100):
            game.step(UP, DOWN)
        self.assertGreaterEqual(game.paddles[0], 3)
        self.assertLessEqual(game.paddles[1], 20)

    def test_expert_is_a_strong_player(self):
        result = match('expert', 'random', games=2, ticks=2000)
        self.assertGreater(result['points'][0], result['points'][1] + 15)
        self.assertGreater(result['return_rate'][0], 0.9)

    def test_expert_anticipates_wall_reflection(self):
        game = Pong()
        game.x, game.y, game.vx, game.vy = 9, 2, -0.7, -0.7
        game.paddles[0] = 3
        self.assertEqual(expert(game), DOWN)
        self.assertEqual(expert(game, 1), STAY)

    def test_recording_is_reproducible_and_labeled(self):
        a = record(100, 42)
        b = record(100, 42)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(a[0].shape, (100, 4, 24, 32))
        self.assertEqual(a[1].shape, (100, 10))
        self.assertTrue(np.isin(a[1], [STAY, UP, DOWN]).all())
        self.assertGreater(np.unique(a[0], axis=0).shape[0], 10)
        packed = pack(a[0])
        np.testing.assert_array_equal(unpack(torch.tensor(packed)).numpy(), a[0])
        np.testing.assert_array_equal(record(100, 42, packed=True)[0], packed)

    def test_transformer_learns_and_roundtrips_without_game_state(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        x, y, _ = record(12, 3)
        model = PongTransformer(dim=16, layers=1)
        x, y = torch.tensor(x), torch.tensor(y)
        inputs = decoder_inputs(y)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
        def loss():
            return torch.nn.functional.cross_entropy(model(x, inputs).flatten(0, 1), y.flatten())
        before = loss().item()
        for _ in range(15):
            optimizer.zero_grad()
            objective = loss()
            objective.backward()
            optimizer.step()
        model.eval()
        self.assertLess(loss().item(), before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(dict(config=model.config, game_config=asdict(Config()), model=model.state_dict()), path)
            restored, config = load(path)
            torch.testing.assert_close(model(x, inputs), restored(x, inputs))
            player = ModelPlayer(restored)
            self.assertIn(player(x[0].numpy()), (STAY, UP, DOWN))
            self.assertAlmostEqual(sum(player.probabilities), 1.0, places=5)
            self.assertEqual(config, Config())

    def test_queue_calls_transformer_only_when_empty(self):
        class CountingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.calls = 0

            def generate(self, observation):
                self.calls += 1
                moves = torch.tensor([[0, 1, 2, 1, 0, 2, 2, 0, 1, 0]])
                return moves, torch.nn.functional.one_hot(moves, 3).float()

        model = CountingModel()
        player = ModelPlayer(model)
        game = Pong(99)
        output = []
        for tick in range(25):
            # A point resets screen history, but must NOT flush pending actions.
            if tick == 3:
                game.x, game.vx = -1, -0.7
            # No observation is even supplied between refills.
            move = player(game.observation() if tick % 10 == 0 else None)
            output.append(move)
            game.step(move, STAY)
            self.assertEqual(model.calls, tick // 10 + 1)
            self.assertEqual(player.position, tick % 10 + 1)
        self.assertEqual(output[:10], output[10:20])
        self.assertEqual(player.refills, 3)

    def test_sequence_targets_are_actual_future_expert_actions(self):
        game = Pong(12)
        game.step(UP, DOWN)
        before = game.observation()
        labels = future_actions(game, tracker)
        simulated = deepcopy(game)
        expected = [[], []]
        for _ in range(10):
            moves = [tracker(simulated, side) for side in range(2)]
            for side in range(2):
                expected[side].append(moves[side])
            simulated.step(*moves)
        self.assertEqual(labels, expected)
        np.testing.assert_array_equal(before, game.observation())
        game.x, game.vx = 0.1, -0.7
        self.assertIsNone(future_actions(game, tracker))

    def test_decoder_is_causal_and_generation_is_exactly_ten(self):
        torch.manual_seed(3)
        model = PongTransformer(dim=16, layers=1).eval()
        x, target, _ = record(2, 8)
        x, target = torch.tensor(x), torch.tensor(target)
        tokens = decoder_inputs(target)
        self.assertTrue((tokens[:, 0] == BOS).all())
        torch.testing.assert_close(tokens[:, 1:], target[:, :-1])
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 1) % 3
        with torch.inference_mode():
            torch.testing.assert_close(model(x, tokens)[:, :5], model(x, changed)[:, :5])
            moves, probabilities = model.generate(x)
        self.assertEqual(moves.shape, (2, 10))
        self.assertEqual(probabilities.shape, (2, 10, 3))
        self.assertTrue(((moves >= 0) & (moves < 3)).all())


if __name__ == '__main__':
    unittest.main()
