import unittest
from .data import generate, advance, END


class MazeTests(unittest.TestCase):
    def test_labels_reach_goal_without_collisions(self):
        for size in (7, 11, 15):
            for seed in range(100):
                grid, pos, goal, path = generate(seed, size)
                self.assertEqual(len(grid), size)
                self.assertEqual(path[-1], END)
                for action in path[:-1]:
                    pos, valid = advance(grid, pos, action)
                    self.assertTrue(valid)
                self.assertEqual(pos, goal)

    def test_causality(self):
        import torch
        from .model import MazeTransformer
        torch.set_num_threads(2)
        model = MazeTransformer(width=32, layers=1).eval()
        grid = torch.tensor([generate(3)[0]])
        with torch.inference_mode():
            a = model(grid, torch.tensor([[5, 0, 1]]))
            b = model(grid, torch.tensor([[5, 3, 2]]))
        torch.testing.assert_close(a[:, 0], b[:, 0])

    def test_wall_collision_does_not_move_player(self):
        grid, _, _, _ = generate(4)
        position, valid = advance(grid, (1, 1), 0)
        self.assertFalse(valid)
        self.assertEqual(position, (1, 1))

    def test_checkpoint_round_trip(self):
        import io
        import torch
        from .model import MazeTransformer, load
        model = MazeTransformer(width=32, layers=1).eval()
        grid = torch.tensor([generate(3)[0]])
        tokens = torch.tensor([[5, 0]])
        buffer = io.BytesIO()
        torch.save(dict(config=model.config, model=model.state_dict()), buffer)
        buffer.seek(0)
        restored = load(buffer)
        with torch.inference_mode():
            torch.testing.assert_close(model(grid, tokens), restored(grid, tokens))


if __name__ == '__main__':
    unittest.main()
