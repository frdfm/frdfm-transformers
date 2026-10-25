import itertools
import unittest
import numpy as np
import torch
from text2image.stick_figures.data import COMBINATIONS, STATES, make_dataset, split_combinations, encode, describe, target
from text2image.stick_figures.model import PoseTransformer
from text2image.stick_figures.render import draw_figure, joints


class ExperimentTests(unittest.TestCase):
    def test_split_has_no_combination_leakage(self):
        data = make_dataset()
        sets = {name: set(map(tuple, rows)) for name, rows in data["combinations"].items()}
        self.assertEqual([len(sets[n]) for n in ("train", "val", "test")], [84,12,12])
        self.assertEqual(set.union(*sets.values()), set(COMBINATIONS))
        for a,b in itertools.combinations(sets,2):
            self.assertFalse(sets[a] & sets[b])
        captions = {name: {r["caption"] for r in records} for name, records in data["records"].items()}
        for a, b in itertools.combinations(captions, 2):
            self.assertFalse(captions[a] & captions[b])
        for name, records in data["records"].items():
            for row in records:
                self.assertIn(tuple(row["combo"]), sets[name])
        for i, states in enumerate(STATES):
            self.assertEqual({c[i] for c in sets["train"]}, set(range(len(states))))
        self.assertTrue(all(not (c[0] == 2 and c[1] == 2) for c in sets["train"]))
        self.assertTrue(all(c[0] == 2 and c[1] == 2 for c in sets["test"]))
        self.assertTrue(all(not (c[0] == 1 and c[2] == 2) for c in sets["train"]))
        self.assertTrue(all(c[0] == 1 and c[2] == 2 for c in sets["val"]))
        self.assertIn((2,2,0,0,0), sets["test"])
        self.assertNotIn((2,2,0,0,0), sets["train"])

    def test_skeleton_and_model_shapes(self):
        torch.set_num_threads(2)
        for combo in COMBINATIONS:
            xy=np.stack(list(joints(target(combo)).values()))
            self.assertTrue(np.isfinite(xy).all())
            self.assertTrue((xy >= 0).all() and (xy[:,0] < 256).all() and (xy[:,1] < 240).all())
        tokens = torch.tensor([encode(describe(c)) for c in COMBINATIONS[:2]])
        model=PoseTransformer(width=16, layers=1, heads=2)
        prediction=model(tokens)
        self.assertEqual(prediction.shape,(2,5))
        prediction.square().mean().backward()
        self.assertIsNotNone(model.embed.weight.grad)
        self.assertEqual(draw_figure(target(COMBINATIONS[0])).size,(256,240))
        with self.assertRaises(ValueError):
            encode("purple elephant")


if __name__ == "__main__":
    unittest.main()
