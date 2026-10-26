"""Dataset expansion must preserve the experiment's evaluation boundary."""
import contextlib,hashlib,io,json,shutil,tempfile,unittest
from pathlib import Path
import numpy as np
from text2image.stick_colors.tokenization import HERE
from text2image.stick_colors.expand_data import expand,sha
from text2image.stick_colors.train import validate_resume_data


class ExpansionTests(unittest.TestCase):
    def test_append_only_expansion_and_resume_guards(self):
        with np.load(HERE/'data/train.npz') as data:
            arrays={key:data[key][:3] for key in data.files}
        captions=json.loads((HERE/'data/train-captions.json').read_text())[:3]
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            np.savez_compressed(folder/'train.npz',**{key:array[:2] for key,array in arrays.items()})
            np.savez_compressed(folder/'val.npz',**{key:array[2:] for key,array in arrays.items()})
            (folder/'train-captions.json').write_text(json.dumps(captions[:2]))
            (folder/'val-captions.json').write_text(json.dumps(captions[2:]))
            shutil.copy2(HERE/'data/tokenizer.json',folder/'tokenizer.json')
            manifest=dict(records={'train':2,'val':1},files_sha256={name:sha(folder/name) for name in ('train.npz','val.npz')},tokenizer_sha256=sha(folder/'tokenizer.json'))
            manifest['dataset_sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
            (folder/'manifest.json').write_text(json.dumps(manifest))
            state=dict(dataset_sha256=manifest['dataset_sha256'],tokenizer_sha256=manifest['tokenizer_sha256'])
            frozen={name:sha(folder/name) for name in ('val.npz','val-captions.json','tokenizer.json')}
            with contextlib.redirect_stdout(io.StringIO()):new=expand(folder,target=3,seed=91)
            with np.load(folder/'train.npz') as data:
                self.assertEqual(len(data['pixels']),3)
                for key,array in arrays.items():np.testing.assert_array_equal(data[key][:2],array[:2])
            self.assertEqual(json.loads((folder/'train-captions.json').read_text())[:2],captions[:2])
            for name,expected in frozen.items():self.assertEqual(sha(folder/name),expected)
            self.assertEqual(new['parent_dataset_sha256'],state['dataset_sha256'])
            with self.assertRaisesRegex(ValueError,'accept-expanded-data'):validate_resume_data(state,new,folder=folder)
            with contextlib.redirect_stdout(io.StringIO()):validate_resume_data(state,new,True,folder)
            wrong=dict(new,parent_dataset_sha256='unrelated')
            with self.assertRaisesRegex(ValueError,'directly descend'):validate_resume_data(state,wrong,True,folder)
            file=folder/'val-captions.json';original=file.read_bytes();file.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'Frozen'):validate_resume_data(state,new,True,folder)
            file.write_bytes(original)
            with np.load(folder/'train.npz') as data:changed={key:data[key] for key in data.files}
            changed['pixels'][0,0]=(int(changed['pixels'][0,0])+1)%7
            np.savez_compressed(folder/'train.npz',**changed)
            new['files_sha256']['train.npz']=sha(folder/'train.npz')
            with self.assertRaisesRegex(ValueError,'Original training rows changed'):validate_resume_data(state,new,True,folder)


if __name__=='__main__':unittest.main()
