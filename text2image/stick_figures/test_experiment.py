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

    def test_factorized_routing_and_clause_permutation(self):
        from text2image.stick_figures.data import encode_clauses, clause_supervision
        from text2image.stick_figures.model import FactorizedPoseTransformer
        from unittest.mock import patch
        torch.set_num_threads(2)
        text = describe((2, 2, 0, 0, 0))
        clauses = text.split(" ; ")
        model = FactorizedPoseTransformer(width=16, layers=1, heads=2).eval()
        tokens = torch.tensor([encode_clauses(text)])
        shuffled = torch.tensor([encode_clauses(" ; ".join(reversed(clauses)))])
        # Inference does not call training-label construction.
        with patch("text2image.stick_figures.data.clause_supervision", side_effect=AssertionError("Training-only labels used")):
            a, identity, angles = model.details(tokens)
            b = model(shuffled)
        self.assertEqual(a.shape, (1,5))
        self.assertEqual(identity.shape, (1,5,5))
        self.assertEqual(angles.shape, (1,5))
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        # Altering another clause cannot change this clause's own local encoding/head outputs.
        changed = torch.tensor([encode_clauses(describe((2, 2, 1, 0, 0)))])
        _, new_identity, new_angles = model.details(changed)
        torch.testing.assert_close(identity[:,0],new_identity[:,0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(angles[:,0],new_angles[:,0],atol=1e-6,rtol=1e-5)
        parts, values = clause_supervision([{"caption":text,"target":target((2,2,0,0,0))}])
        self.assertEqual(parts, [[0,1,2,3,4]])
        self.assertEqual(values, [[1.0,1.0,0.0,0.0,0.0]])
        (a.square().mean() + identity.square().mean() + angles.square().mean()).backward()
        self.assertIsNotNone(model.embed.weight.grad)
        with self.assertRaises(ValueError):
            encode_clauses("left arm raised")

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
