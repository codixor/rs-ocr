"""Isolated model preparation and CPU reference; never used by the Rust API."""
import argparse
import hashlib
import json
from pathlib import Path

CRATE=Path(__file__).resolve().parent.parent
MODEL=CRATE/'models/pp-doclayout-s'
MANIFEST=CRATE/'model-manifests/doc-layout-model.json'


def prepare():
    import numpy as np
    import onnx
    from onnx import numpy_helper
    spec=json.loads(MANIFEST.read_text())
    src=MODEL/'source.onnx'
    if hashlib.sha256(src.read_bytes()).hexdigest()!=spec['artifacts'][0]['sha256']:
        raise ValueError('source ONNX checksum mismatch')
    model=onnx.load(src)
    nms=[n for n in model.graph.node if n.op_type=='NonMaxSuppression']
    if len(nms)!=1:raise ValueError('expected exactly one native class-aware NMS')
    before=[i.SerializeToString() for i in model.graph.initializer]
    for index,old,new in [(3,0.6,0.5),(4,0.025,0.3)]:
        name=nms[0].input[index]
        node=next(n for n in model.graph.node if name in n.output)
        attribute=next(a for a in node.attribute if a.name=='value')
        array=numpy_helper.to_array(attribute.t)
        if array.dtype!=np.float32 or array.shape!=(1,) or abs(float(array[0])-old)>1e-6:
            raise ValueError('unexpected source NMS constant')
        attribute.t.CopyFrom(numpy_helper.from_array(np.array([new],dtype=np.float32)))
    if before!=[i.SerializeToString() for i in model.graph.initializer]:
        raise ValueError('model weights changed')
    onnx.checker.check_model(model)
    dest=MODEL/'pp_doclayout_s.onnx';onnx.save(model,dest)
    if hashlib.sha256(dest.read_bytes()).hexdigest()!=spec['prepared_sha256']:
        raise ValueError('non-reproducible prepared graph')
    print('DOC_LAYOUT_PREPARED')


def reference(jobs_path,output_path):
    import cv2
    import numpy as np
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1;opts.log_severity_level=3
    session=ort.InferenceSession(str(MODEL/'pp_doclayout_s.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    output=[]
    for job in json.loads(Path(jobs_path).read_text()):
        image=cv2.imdecode(np.frombuffer(Path(job['image']).read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('reference image decode failed')
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);height,width=image.shape[:2]
        pixels=cv2.resize(image,(480,480),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255
        pixels=(pixels-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        rows,count=session.run(None,{'image':np.transpose(pixels,(2,0,1))[None],
            'scale_factor':np.array([[480/height,480/width]],dtype=np.float32)})
        rows=rows[:int(count[0])];rows=rows[rows[:,1]>=.5]
        regions=[{'class_id':int(r[0]),'confidence':float(r[1]),'raw_bounds':r[2:].tolist()} for r in rows]
        output.append({'id':job['id'],'width':width,'height':height,'regions':regions})
    result={'runtime':{'onnxruntime':ort.__version__,'opencv':cv2.__version__,'numpy':np.__version__,'provider':'CPUExecutionProvider','threads':1},'predictions':output}
    Path(output_path).write_text(json.dumps(result,indent=2)+'\n')


def fixture():
    import cv2
    import numpy as np
    width,height=997,541
    y,x=np.indices((height,width));image=np.stack([(x*13+y*7)%256,(x*3+y*17)%256,(x*29+y*5)%256],axis=2).astype(np.uint8)
    scaled=cv2.resize(image,(480,480),interpolation=cv2.INTER_CUBIC)
    points=[(0,0),(1,2),(17,43),(100,200),(239,239),(301,411),(478,479),(479,479)]
    value={'source_width':width,'source_height':height,'generator':'RGB=((13*x+7*y)%256,(3*x+17*y)%256,(29*x+5*y)%256)',
        'reference':'OpenCV '+cv2.__version__+' INTER_CUBIC, target480x480, half-pixel coordinates',
        'points':[{'x':x,'y':y,'rgb':scaled[y,x].tolist()} for x,y in points]}
    (CRATE/'tests/fixtures/doc_layout_cubic.json').write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','reference','fixture'));p.add_argument('--jobs');p.add_argument('--output');a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='fixture':fixture()
    else:
        if not a.jobs or not a.output:p.error('reference requires --jobs and --output')
        reference(a.jobs,a.output)


if __name__=='__main__':main()
