"""Behavioral controls for historical binding and fresh layout measurements."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import financial_benchmark as f
import layout_benchmark as b
import ocr_model_study as model


class LayoutBenchmarkTests(unittest.TestCase):
    def test_history_accepts_only_original_current_or_archived_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'root';archive=Path(directory)/'archive'
            root.mkdir();archive.mkdir()
            sha=hashlib.sha256(b'original').hexdigest()
            (root/'source.rs').write_bytes(b'original')
            f.verify_sources({'source.rs':sha},root=root,archive=archive)
            (root/'source.rs').write_bytes(b'improved')
            with self.assertRaisesRegex(ValueError,'frozen source'):f.verify_sources({'source.rs':sha},root=root,archive=archive)
            (archive/'source.rs').write_bytes(b'original')
            f.verify_sources({'source.rs':sha},root=root,archive=archive)
            (archive/'source.rs').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'frozen source'):f.verify_sources({'source.rs':sha},root=root,archive=archive)
            with self.assertRaisesRegex(ValueError,'unsafe'):f.verify_sources({'../outside':sha},root=root,archive=archive)

    def test_archive_does_not_allow_running_old_study_with_new_parser(self):
        with patch.object(f,'code_hashes',return_value={'source':'changed'}):
            with self.assertRaisesRegex(ValueError,'changed after freeze'):f.execute({},'test')
            with self.assertRaisesRegex(ValueError,'start a new study'):model.run('tiny','test',[])

    def test_freshness_detects_old_id_or_identical_image_under_an_alias(self):
        manifest=f.read(b.MANIFEST)
        prior=[f.read(f.MANIFEST),f.read(model.SAMPLES)]
        b.validate(manifest,prior)
        sample=next(s for s in manifest['samples'] if s['split']=='test')
        old=copy.deepcopy(prior)
        old[0]['samples'].append(sample)
        with self.assertRaisesRegex(ValueError,'prior study'):b.validate(manifest,old)
        alias=copy.deepcopy(sample);alias['id']='different-id'
        old=copy.deepcopy(prior);old[0]['samples'].append(alias)
        with self.assertRaisesRegex(ValueError,'prior study'):b.validate(manifest,old)

    def test_data_verifier_rejects_changed_checksums(self):
        manifest=f.read(b.MANIFEST); manifest['samples'][0]['artifacts'][0]['sha256']='0'*64
        original=f.read
        with patch.object(f,'read',side_effect=lambda p:manifest if p==b.MANIFEST else original(p)):
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):b.verify_data()

    def test_candidate_evidence_and_confidence_are_independently_checked(self):
        samples=[{'id':'a'}]
        row={'spans':[{'confidence':0.8},{'confidence':0.2}],'fields':{'total':{'candidates':[{'span_indices':[0,1],'confidence':0.2}]}}}
        b.check_rows(samples,{'a':row})
        wrong=copy.deepcopy(row);wrong['fields']['total']['candidates'][0]['confidence']=0.8
        with self.assertRaisesRegex(ValueError,'weakest'):b.check_rows(samples,{'a':wrong})
        wrong=copy.deepcopy(row);wrong['fields']['total']['candidates'][0]['span_indices']=[2]
        with self.assertRaisesRegex(ValueError,'evidence'):b.check_rows(samples,{'a':wrong})
        with self.assertRaisesRegex(ValueError,'incomplete'):b.check_rows(samples,{})

    def test_cannot_refreeze_after_heldout_output_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory);(data/'improved-test.jsonl').write_text('{}\n')
            with patch.object(b,'DATA',data):
                with self.assertRaisesRegex(ValueError,'cannot refreeze'):b.freeze()

    def test_test_execution_refuses_changed_frozen_source(self):
        with patch.object(f,'read',return_value={'sources':{}}),patch.object(b,'sources',return_value={'new':'sha'}):
            with self.assertRaisesRegex(ValueError,'changed'):b.run('test')


if __name__=='__main__':unittest.main()
