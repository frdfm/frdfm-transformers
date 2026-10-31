from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .demo import projected_path
from .evaluate import evaluate
from .game import Config, LEFT, RIGHT, STRAIGHT, Session, Snake, expert
from .planning import PlanGeometry
from .model import BOS, HORIZON, ModelPlayer, SnakeTransformer, load
from .prepare_data import future_actions, record
from .train import decoder_inputs, validate


class SnakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_history_heading_turns_and_growth(self):
        game = Snake(2)
        before = game.observation()
        self.assertEqual(before.shape, (2, 20, 20))
        self.assertEqual(int(before[-1][game.body[0]]), 4 + game.direction)
        destination, direction = game.destination(LEFT)
        game.food = destination
        self.assertTrue(game.step(LEFT))
        self.assertEqual(game.body[0], destination)
        self.assertEqual(game.direction, direction)
        self.assertEqual((game.score, len(game.body)), (1, 4))
        self.assertNotIn(game.food, game.body)
        np.testing.assert_array_equal(game.observation()[0], before[-1])
        copy = game.observation()
        copy[:] = 0
        self.assertGreater(game.observation().sum(), 0)
        self.assertEqual(game.destination(RIGHT)[1], (direction + 1) % 4)

    def test_wall_body_tail_and_full_board(self):
        game = Snake()
        game.body, game.direction, game.food = [(1, 2), (2, 2), (3, 2)], 0, (4, 4)
        self.assertFalse(game.step(STRAIGHT))
        self.assertFalse(game.alive)
        ticks = game.ticks
        game.step(LEFT)
        self.assertEqual(game.ticks, ticks)
        game = Snake()
        game.body, game.direction, game.food = [(3, 3), (3, 2), (4, 2), (4, 3)], 1, (8, 8)
        self.assertTrue(game.safe(RIGHT))  # Tail vacates on the same tick.
        game.step(RIGHT)
        self.assertEqual(game.body[0], (4, 3))
        game.body = [(4, 3), (3, 3), (3, 2), (4, 2), (5, 2)]
        self.assertFalse(game.safe(RIGHT))  # This cell is no longer the tail.
        game.body = [(r, c) for r in range(1, 19) for c in range(1, 19)]
        self.assertIsNone(game.spawn_food())
        with self.assertRaises(ValueError):
            game.destination(3)
        with self.assertRaises(ValueError):
            Config(height=4)

        game = Snake(config=Config(height=8, width=8))
        game.food, game.direction = (1, 2), 1
        game.body = [(1, 1)] + [(r, c) for r in range(1, 7) for c in range(1, 7)
                                  if (r, c) not in ((1, 1), game.food)]
        self.assertTrue(game.step(STRAIGHT))
        self.assertTrue(game.won)
        self.assertIsNone(game.food)
        self.assertEqual(len(game.body), 36)
        game.step(STRAIGHT)
        self.assertEqual(game.ticks, 1)

    def test_targets_follow_teacher_and_continue_safely_after_food(self):
        game = Snake(5)
        before, rng = game.observation(), game.rng.getstate()
        expected = []
        future = deepcopy(game)
        for _ in range(10):
            move = expert(future)
            if not future.safe(move):
                break
            expected.append(move)
            if future.step(move):
                future.food = None
        target = future_actions(game)
        np.testing.assert_array_equal(target[:len(expected)], expected)
        self.assertTrue((target[len(expected):] == -100).all())
        np.testing.assert_array_equal(game.observation(), before)
        self.assertEqual(game.rng.getstate(), rng)
        game.food = game.destination(STRAIGHT)[0]
        target = future_actions(game)
        self.assertEqual(target[0], STRAIGHT)
        self.assertTrue((target >= 0).all())
        simulated = deepcopy(game)
        for move in target:
            self.assertTrue(simulated.safe(move))
            if simulated.step(move):
                simulated.food = None
        # The continuation must not depend on the hidden random food sequence.
        game.rng.seed(987654)
        np.testing.assert_array_equal(future_actions(game), target)

    def test_recording_is_reproducible(self):
        a, b = record(24, 42), record(24, 42)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(a[1].shape, (24, 10))
        self.assertTrue(np.isin(a[1], [-100, 0, 1, 2]).all())
        self.assertTrue((a[1][:, 0] >= 0).all())

    def test_causality_shift_exact_horizon_and_learning(self):
        torch.manual_seed(3)
        x, y, _ = record(8, 5)
        x, y = torch.tensor(x), torch.tensor(y)
        tokens = decoder_inputs(y)
        self.assertTrue((tokens[:, 0] == BOS).all())
        torch.testing.assert_close(tokens[:, 1:], y[:, :-1].clamp_min(0))
        model = SnakeTransformer(dim=16, layers=1).eval()
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 1) % 3
        with torch.inference_mode():
            torch.testing.assert_close(model(x, tokens)[:, :5], model(x, changed)[:, :5])
            moves, probabilities = model.generate(x)
        self.assertEqual(moves.shape, (8, HORIZON))
        self.assertEqual(probabilities.shape, (8, HORIZON, 3))
        self.assertTrue(((moves >= 0) & (moves < 3)).all())
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(8, HORIZON))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
        def loss():
            return torch.nn.functional.cross_entropy(model(x, tokens).flatten(0, 1), y.flatten())
        before = loss().item()
        for _ in range(20):
            optimizer.zero_grad()
            objective = loss()
            objective.backward()
            optimizer.step()
        self.assertLess(loss().item(), before * 0.7)
        report = validate(model, x, y)
        self.assertEqual(report['position_counts'], (y != -100).sum(0).tolist())
        self.assertTrue(0 <= report['generated_accuracy'] <= 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(dict(experiment='snake-seq10', config=model.config,
                            game_config=asdict(Config()), model=model.state_dict()), path)
            restored, config = load(path)
            self.assertEqual(config, Config())
            torch.testing.assert_close(model(x, tokens), restored(x, tokens))

    def test_queue_is_not_refilled_after_eating(self):
        class CountingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.calls = 0

            def generate(self, x):
                self.calls += 1
                moves = torch.zeros(1, 10, dtype=torch.long)
                return moves, torch.nn.functional.one_hot(moves, 3).float()

        model = CountingModel()
        player = ModelPlayer(model)
        game = Snake()
        game.food = game.destination(STRAIGHT)[0]
        game.step(player(game.observation()))
        self.assertEqual(game.score, 1)
        for tick in range(1, 25):
            player(game.observation() if player.needs_plan else None)
            self.assertEqual(model.calls, tick // 10 + 1)
        self.assertEqual(player.position, 5)
        player.reset()
        self.assertTrue(player.needs_plan)
        with self.assertRaises(ValueError):
            player()
        # Actual matches stop at death; each living tick still consumes one token.
        with patch('snake.evaluate.controller', return_value=(player, Config())):
            result = evaluate('model', games=3, ticks=35)
        for episode in result['episodes']:
            self.assertEqual(episode['refills'], (episode['ticks'] + 9) // 10)

    def test_preview_is_display_only_and_continues_after_meal(self):
        game = Snake(42)
        game.food = game.destination(STRAIGHT)[0]
        before = deepcopy(game)
        path = projected_path(game, [STRAIGHT] * 10)
        self.assertGreater(len(path), 1)
        self.assertEqual(game.body, before.body)
        self.assertEqual(game.rng.getstate(), before.rng.getstate())
        np.testing.assert_array_equal(game.observation(), before.observation())

    def test_restart_randomizes_and_seed_reproduces_the_session(self):
        a, b = Session(seed=42), Session(seed=42)
        starts = []
        for _ in range(5):
            self.assertEqual(a.seed, b.seed)
            np.testing.assert_array_equal(a.game.observation(), b.game.observation())
            starts.append((a.game.body[0], a.game.direction, a.game.food))
            a.restart()
            b.restart()
        self.assertEqual(len(set(starts)), 5)
        self.assertNotEqual(Session().seed, Session().seed)

    def test_corner_food_plans_do_not_run_into_walls(self):
        # Force logits to prefer FORWARD at every position, including after food.
        model = SnakeTransformer(dim=16, layers=1).eval()
        for parameter in model.parameters():
            parameter.data.zero_()
        model.head[-1].bias.data[STRAIGHT] = 10
        games = []
        c = Config()
        for row, col, direction in ((1, 1, 3), (1, c.width-2, 0),
                                     (c.height-2, 1, 2), (c.height-2, c.width-2, 1)):
            game = Snake(config=c)
            dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
            game.direction = direction
            game.food = (row, col)
            game.body = [(row-dr*i, col-dc*i) for i in range(1, 4)]
            game.frames.clear()
            game.frames.extend(game.matrix() for _ in range(c.history))
            games.append(game)
        x = torch.tensor(np.stack([g.observation() for g in games]))
        moves, probabilities = model.generate(x)
        for game, plan in zip(games, moves.tolist()):
            for move in plan:
                self.assertTrue(game.safe(move))
                game.step(move)
            self.assertEqual(game.score, 1)
            self.assertTrue(game.alive)
        # The unsafe preferred action is removed rather than executed.
        self.assertEqual(probabilities[0, 1, STRAIGHT].item(), 0)

    def test_planning_masks_choose_a_complete_path_over_a_dead_end(self):
        game = Snake()
        game.body, game.direction, game.food = [(2, 1), (3, 1), (4, 1)], 0, (1, 1)
        geometry = PlanGeometry(game.matrix())
        mask = geometry.allowed(10)
        self.assertTrue(mask[STRAIGHT])  # Eat in corner, then turn along top edge.
        geometry.advance(STRAIGHT)
        self.assertFalse(geometry.allowed(9)[STRAIGHT])
        self.assertTrue(geometry.allowed(9)[RIGHT])
        # Immediate forward is free, but the next cell is a cul-de-sac.
        geometry = PlanGeometry(game.matrix())
        geometry.free = {(1, 1), (2, 2), (2, 3), (2, 4), (3, 4), (4, 4)}
        self.assertEqual(geometry.allowed(4), [False, False, True])


if __name__ == '__main__':
    unittest.main()
