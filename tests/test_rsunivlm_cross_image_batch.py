"""WSL-only synthetic cross-image batching checks; no real model weights."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from baselines.rsunivlm import evaluate as legacy
from baselines.rsunivlm import protocols as protocol
from tools import rsunivlm_batch_runtime as runtime
from tools.rsunivlm_batch_prefetch import CrossImagePrefetch
from test_rsunivlm_protocols import rle


def sources(tmp_path, count=5):
    values=[]
    tmp_path.mkdir(parents=True,exist_ok=True)
    for index in range(count):
        image=Image.new("RGB",(1024,1024))
        for tile in range(4):
            x,y=(tile%2)*512,(tile//2)*512
            image.paste((index*4+tile,0,0),(x,y,x+512,y+512))
        path=tmp_path/f"s{index}.png"
        image.save(path)
        label=tmp_path/f"s{index}_gt.png"
        label.write_bytes(b"GT pixels must never be decoded by prefetch")
        values.append(protocol.Sample(f"s{index}",(path,),label))
    return values


class Bulk:
    requested_batch_size=effective_batch_size=16
    format_retries=2
    verification_seconds=0.
    oom_splits=0
    def __init__(self,bad_red=None):
        self.calls=[]
        self.bad_red=bad_red
    def generate_batch(self,images,prompt):
        colors=[image.getpixel((0,0))[0] for image in images]
        self.calls.append(colors)
        return [dict(response="water" if red==self.bad_red else rle(f"{protocol.LOVEDA_CLASSES[red%7]}*24"),
            generation=dict(ended_with_eos=True,hit_token_limit=False,inference_batch_size=len(images)))
            for red in colors]


def item_id(sample,config="fixture"):
    return protocol.fingerprint([config,protocol.input_identity(sample)])


def reconstruct(prefetch,backend,sample,output):
    with Image.open(sample.images[0]) as opened:
        image=opened.convert("RGB")
    info=prefetch.prepare(backend,image,sample,output/"raw_text",item_id(sample),tile_size=512,overwrite=False)
    prediction,_=runtime.predict_tiled(backend,image,sample,output/"raw_text",item_id(sample),tile_size=512,overwrite=False)
    image.close()
    return prediction,info


def assert_quadrants(prediction,index):
    for tile in range(4):
        x,y=(tile%2)*512,(tile//2)*512
        assert np.all(prediction[y:y+512,x:x+512]==(index*4+tile)%7)


def test_four_images_really_dispatch_sixteen_tiles_and_restore_separately(tmp_path,monkeypatch):
    data=sources(tmp_path/"data")
    output=tmp_path/"output"
    labels={sample.label for sample in data}
    original_open=Image.open
    def rgb_only(path,*args,**kwargs):
        assert Path(path) not in labels,"lookahead read GT pixels"
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Image,"open",rgb_only)
    backend=Bulk()
    prefetch=CrossImagePrefetch(data,"fixture")
    infos=[]
    for index,sample in enumerate(data):
        prediction,info=reconstruct(prefetch,backend,sample,output)
        assert_quadrants(prediction,index)
        infos.append(info)
    assert [len(call) for call in backend.calls]==[16,4]
    assert backend.calls[0]==list(range(16)) and backend.calls[1]==list(range(16,20))
    assert backend.actual_batch_sizes=={16,4}
    assert infos[0]["prefetch_source_images"]==4 and infos[0]["prefetch_tile_queries"]==16
    assert infos[1]["prefetch_wall_seconds_charged"]==0
    assert infos[4]["prefetch_source_images"]==1 and infos[4]["prefetch_tile_queries"]==4
    for index,sample in enumerate(data[:4]):
        for tile in range(4):
            x,y=(tile%2)*512,(tile//2)*512
            record=legacy.read_json(output/"raw_text"/f"{sample.name}_tile512_y{y}_x{x}_seg.json")
            generation=record["generation"]
            assert generation["inference_batch_size"]==16
            assert len(generation["batch_members"])==16
            assert generation["batch_position"]==index*4+tile
            assert generation["batch_members"][index*4+tile]==record["identity"]
    # New process, same raw identities: rebuild without another model call.
    resumed=Bulk()
    prediction,info=reconstruct(CrossImagePrefetch(data,"fixture"),resumed,data[0],output)
    assert_quadrants(prediction,0)
    assert not resumed.calls and info["prefetch_reused_queries"]==4


def test_future_format_failure_does_not_prevent_an_earlier_valid_source_from_scoring(tmp_path):
    data=sources(tmp_path/"data",4)
    backend=Bulk(bad_red=4)  # first tile of the second source, not current source
    prefetch=CrossImagePrefetch(data,"fixture")
    output=tmp_path/"output"
    prediction,info=reconstruct(prefetch,backend,data[0],output)
    assert_quadrants(prediction,0)
    assert [len(call) for call in backend.calls]==[16,1,1]
    with pytest.raises(ValueError,match="remain invalid"):
        reconstruct(prefetch,backend,data[1],output)
    failed=legacy.read_json(output/"raw_text"/"s1_tile512_y0_x0_seg.json")
    assert len(failed["attempts"])==3 and all(row["response"]=="water" for row in failed["attempts"])


def test_cross_image_oom_halves_batch_without_dropping_or_reordering_tiles(tmp_path):
    class Limited(Bulk):
        def generate_batch(self,images,prompt):
            if len(images)>8:
                raise torch.cuda.OutOfMemoryError("synthetic batch16 OOM")
            return super().generate_batch(images,prompt)
    data=sources(tmp_path/"data",4)
    backend=Limited()
    prefetch=CrossImagePrefetch(data,"fixture")
    for index,sample in enumerate(data):
        prediction,_=reconstruct(prefetch,backend,sample,tmp_path/"output")
        assert_quadrants(prediction,index)
    assert [len(call) for call in backend.calls]==[8,8]
    assert backend.oom_splits==1 and backend.effective_batch_size==8
    assert backend.actual_batch_sizes=={8}


def test_prefetch_skips_a_verified_cached_png_without_reading_its_gt(tmp_path):
    data=sources(tmp_path/"data",5)
    output=tmp_path/"output"
    prediction=np.zeros((1024,1024),np.uint8)
    path=output/"pred_mask"/"s1_pred_mask.png"
    path.parent.mkdir(parents=True)
    Image.fromarray(prediction).save(path)
    legacy.write_json(output/"receipts"/"s1.json",dict(identity=item_id(data[1]),
        prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest()))
    backend=Bulk()
    reconstructed,_=reconstruct(CrossImagePrefetch(data,"fixture"),backend,data[0],output)
    assert_quadrants(reconstructed,0)
    assert backend.calls==[list(range(4))+list(range(8,20))]


def test_sixteen_sequence_model_call_preserves_images_and_individual_eos():
    class Template:
        roles=("user","assistant")
        def append_message(self,*args): pass
        def get_prompt(self): return "same prompt"
    def forward(tokens,**kwargs):
        assert tokens.shape==(16,3) and kwargs["modalities"]==["image"]*16
        assert len(kwargs["images"])==16 and kwargs["granularity"]==2
        assert [int(image[0,0,0]) for image in kwargs["images"]]==list(range(16))
        output=torch.full((16,4),99,dtype=torch.long)
        for index in range(16):
            output[index,0]=index+10
            if index%2: output[index,1]=index+100
        return output
    core=SimpleNamespace(device=torch.device("cpu"),dtype=torch.float32,context_len=8192,max_new_tokens=20,
        template=Template(),processor=None,image_token_index=-200,
        tokenizer=SimpleNamespace(eos_token_id=99,batch_decode=lambda rows,**kwargs:[str(rows[0][0])]),
        tokenizer_image_token=lambda *args,**kwargs:torch.tensor([2,-200,3]),
        process_images=lambda images,*args:torch.stack([torch.full((3,4,4),image.getpixel((0,0))[0]) for image in images]),
        model=SimpleNamespace(config=SimpleNamespace(image_aspect_ratio="square"),
                              generation_config=SimpleNamespace(eos_token_id=99),generate=forward))
    images=[Image.new("RGB",(512,512),(index,0,0)) for index in range(16)]
    answers=runtime.generate_batch(core,images,protocol.LOVEDA_PROMPT,verify_processor=True)
    assert [answer["response"] for answer in answers]==[str(index+10) for index in range(16)]
    assert [answer["generation"]["sequence_tokens"] for answer in answers]==[2 if index%2==0 else 3 for index in range(16)]
    assert all(answer["generation"]["inference_batch_size"]==16 for answer in answers)
