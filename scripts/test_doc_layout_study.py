"""Controls for optional layout provenance, CPU parity and safe backfill scoring."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import doc_layout_study as b


class DocLayoutStudyTests(unittest.TestCase):
    def parity(self):
        refs=[{'id':str(i),'width':100,'height':200,'regions':([] if i==3 else [{'class_id':2,'confidence':.8,'raw_bounds':[1,2,30,40]}])} for i in range(4)]
        native={r['id']:{'layout':dict(r,regions=[dict(o,class_name='text') for o in r['regions']])} for r in refs}
        return native,{'predictions':refs}

    def test_positive_and_negative_parity_controls_and_wrong_predictions(self):
        native,ref=self.parity();self.assertEqual(len(b.compare_parity(native,ref)),4)
        for bad in ('class','score','coordinates','missing','all_empty'):
            n=copy.deepcopy(native)
            if bad=='class':n['0']['layout']['regions'][0]['class_id']=9
            elif bad=='score':n['0']['layout']['regions'][0]['confidence']=.6
            elif bad=='coordinates':n['0']['layout']['regions'][0]['raw_bounds'][0]=10
            elif bad=='missing':n.pop('0')
            else:
                for row in n.values():row['layout']['regions']=[]
            with self.assertRaises((ValueError,KeyError)):b.compare_parity(n,ref)

    def test_model_bytes_are_checked_against_pinned_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory);(model/'source.onnx').write_bytes(b'corrupt')
            with patch.object(b,'MODEL',model):
                with self.assertRaisesRegex(ValueError,'checksum'):b.verify_artifacts()

    def test_non_finite_wrong_shape_and_duplicate_reference_evidence_is_rejected(self):
        for bad in ('nan_score','nan_box','short_box','boolean_class','duplicate_reference'):
            native,ref=self.parity()
            r=native['0']['layout']['regions'][0]
            if bad=='nan_score':r['confidence']=float('nan')
            elif bad=='nan_box':r['raw_bounds'][0]=float('inf')
            elif bad=='short_box':r['raw_bounds'].pop()
            elif bad=='boolean_class':r['class_id']=True
            else:ref['predictions'].append(copy.deepcopy(ref['predictions'][0]))
            with self.assertRaises(ValueError):b.compare_parity(native,ref)

    def test_old_ids_and_images_cannot_be_fresh_test(self):
        manifest=b.f.read(b.MANIFEST)
        prior=[b.f.read(b.old.MANIFEST)]
        b.old.validate(manifest,prior)
        sample=next(s for s in manifest['samples'] if s['split']=='test')
        overlap=copy.deepcopy(prior);overlap[0]['samples'].append(sample)
        with self.assertRaisesRegex(ValueError,'prior study'):b.old.validate(manifest,overlap)

    def test_history_verifier_detects_rewritten_report(self):
        original=b.f.read
        with patch.object(b.f,'read',side_effect=lambda p:{'docs/layout-results.json':'0'*64} if p==b.HISTORY else original(p)):
            with self.assertRaisesRegex(ValueError,'historical'):b.verify_history()

    def test_no_refreeze_or_test_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory);(data/'test.jsonl').write_text('{}\n')
            with patch.object(b,'DATA',data):
                with self.assertRaisesRegex(ValueError,'refreeze'):b.freeze()

    def test_frozen_execution_rejects_changed_sources_for_both_splits(self):
        with patch.object(b,'verify_data'),patch.object(b,'verify_artifacts'), \
                patch.object(b,'sources',return_value={'new':'sha'}):
            for split in ('dev','test'):
                with self.assertRaisesRegex(ValueError,'implementation/data changed'):b.run(split)

    def test_existing_page_candidates_are_immutable_and_backfills_need_region_evidence(self):
        field={'value':1000,'candidates':[{'value':1000,'raw_text':'10.00','confidence':.3,'span_indices':[0]}]}
        row={'id':'a','spans':[{'confidence':.3}],'layout':{'regions':[{}],'region_spans':[[0]]},
             'baseline':{'fields':{'total':copy.deepcopy(field)},'issues':''},
             'assisted':{'fields':{'total':copy.deepcopy(field)},'issues':''}}
        b.check_results([{'id':'a'}],{'a':row})
        bad=copy.deepcopy(row);bad['assisted']['fields']['total']['value']=2000
        with self.assertRaisesRegex(ValueError,'overwritten'):b.check_results([{'id':'a'}],{'a':bad})
        row['baseline']['fields']['total']={'value':None,'candidates':[]}
        with self.assertRaisesRegex(ValueError,'unreviewed'):b.check_results([{'id':'a'}],{'a':row})
        row['assisted']['issues']='PP-DocLayout-S region association';b.check_results([{'id':'a'}],{'a':row})
        row['layout']['region_spans']=[[]]
        with self.assertRaisesRegex(ValueError,'containing region'):b.check_results([{'id':'a'}],{'a':row})


if __name__=='__main__':unittest.main()
