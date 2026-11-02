"""WSL-only batching/recovery checks using synthetic files and a fake GMoE."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from baselines.rsunivlm import evaluate as legacy
from baselines.rsunivlm import protocols as protocol
from baselines.rsunivlm.model import RSUniVLM
from tools.eval_rsunivlm_batched import alias_decoder, BatchBackend
from tools.rsunivlm_batch_import import build_protocol, import_raw_answers, resolve_output_root, changed_fields
from tools import rsunivlm_batch_runtime as batch
from test_rsunivlm_protocols import FakeModel, args, make_loveda, rle


class FakeBatch:
    format_retries = 2
    requested_batch_size = effective_batch_size = 4
    verification_seconds = 0.
    oom_splits = 0
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def generate_batch(self, images, prompt):
        self.calls.append((list(images), prompt))
        return [dict(response=next(self.responses), generation=dict(ended_with_eos=True,
            hit_token_limit=False, inference_batch_size=len(images))) for image in images]


def jobs(tmp_path, count=4):
    spatial = dict(source_box=[0,0,512,512], source_extent=[512,512], source_tile_size=512,
                   padding="black_right_bottom", valid_height=512, valid_width=512)
    return [dict(image=Image.new("RGB",(512,512),(i*30,0,0)), path=tmp_path/f"tile{i}.json",
                 identity=f"id{i}", spatial=spatial) for i in range(count)]


def test_bareland_is_an_explicit_synonym_but_unknown_names_remain_invalid():
    original = protocol.label_map
    with alias_decoder():
        for word in ("bareland", "bare land", "bare_land", "bare-land"):
            np.testing.assert_array_equal(protocol.decode_rle(rle(f"{word}*24"),"loveda",grid_size=None),
                                          np.full((24,24),4,np.uint8))
        for bad in ("class*24", "remote-sensing*24", "water"):
            with pytest.raises(ValueError):
                protocol.decode_rle(bad,"loveda",grid_size=None)
    assert protocol.label_map is original


def test_four_requests_remain_separate_and_only_invalid_subset_is_retried(tmp_path):
    pending = jobs(tmp_path)
    model = FakeBatch([rle("building*24"),"water",rle("bareland*24"),"class*24",rle("road*24"),rle("water*24")])
    with alias_decoder():
        grids, reused = batch.query_tiles(model,pending)
        assert [len(images) for images,_ in model.calls] == [4,2]
        assert not any(reused.values())
        assert [int(grids[job["path"]][0,0]) for job in pending] == [1,2,4,3]
        assert model.calls[1][0] == [pending[1]["image"],pending[3]["image"]]
        cached, reused = batch.query_tiles(FakeBatch([]),pending)
        assert all(reused.values())
        assert all(np.array_equal(cached[key],grids[key]) for key in grids)


def test_alias_recovers_the_earliest_valid_stored_attempt_without_model_calls(tmp_path):
    job = jobs(tmp_path,1)[0]
    attempts = [dict(prompt_variant=0,prompt=protocol.LOVEDA_PROMPT,response="water"),
                dict(prompt_variant=1,prompt=protocol.format_retry_prompt("loveda",1),response=rle("bareland*24")),
                dict(prompt_variant=2,prompt=protocol.format_retry_prompt("loveda",2),response=rle("forest*24"))]
    legacy.write_json(job["path"],dict(identity=job["identity"],attempts=attempts,**attempts[-1]))
    model = FakeBatch([])
    with alias_decoder():
        grids,_ = batch.query_tiles(model,[job])
    assert not model.calls and int(grids[job["path"]][0,0]) == 4
    record = json.loads(job["path"].read_text())
    assert record["selected_prompt_variant"] == 1 and len(record["attempts"]) == 3
    assert record["format_retries_used"] == 2  # actual historical retry cost


def test_oom_splitting_keeps_request_order_and_does_not_drop_a_sample(tmp_path):
    class OOM(FakeBatch):
        def generate_batch(self, images, prompt):
            if len(images)>2:
                raise torch.cuda.OutOfMemoryError("synthetic batch4 OOM")
            return super().generate_batch(images,prompt)
    model=OOM([rle("building*24")]*4)
    pending=jobs(tmp_path)
    grids,_=batch.query_tiles(model,pending)
    assert len(grids)==4 and model.oom_splits==1 and model.effective_batch_size==2
    assert [len(images) for images,_ in model.calls]==[2,2]


def test_batched_generation_uses_one_image_per_sequence_and_individual_eos():
    core=RSUniVLM.__new__(RSUniVLM)
    core.device,core.dtype=torch.device("cpu"),torch.float32
    core.context_len,core.max_new_tokens,core.image_token_index=8192,20,-200
    class Template:
        roles=("user","assistant")
        def append_message(self,*args): pass
        def get_prompt(self): return "fixed prompt"
    core.template,core.processor=Template(),None
    core.tokenizer=SimpleNamespace(eos_token_id=9,batch_decode=lambda rows,**kwargs:[str(rows[0])])
    core.tokenizer_image_token=lambda *args,**kwargs:torch.tensor([2,-200,3])
    core.process_images=lambda images,*args:torch.stack([torch.full((3,4,4),i.getpixel((0,0))[0]) for i in images])
    captured={}
    def generate(tokens,**kwargs):
        captured.update(kwargs)
        assert tokens.tolist()==[[2,-200,3],[2,-200,3]]
        assert len(kwargs["images"])==2 and all(v.ndim==3 for v in kwargs["images"])
        assert kwargs["modalities"]==["image","image"] and kwargs["granularity"]==2
        assert kwargs["images"][0][0,0,0].item()==10 and kwargs["images"][1][0,0,0].item()==90
        return torch.tensor([[5,9,9,9],[7,8,6,9]])
    core.model=SimpleNamespace(config=SimpleNamespace(image_aspect_ratio="square"),
        generation_config=SimpleNamespace(eos_token_id=9),generate=generate)
    images=[Image.new("RGB",(512,512),(10,0,0)),Image.new("RGB",(512,512),(90,0,0))]
    result=batch.generate_batch(core,images,protocol.LOVEDA_PROMPT,verify_processor=True)
    assert [row["generation"]["sequence_tokens"] for row in result]==[2,4]
    assert all(row["generation"]["inference_batch_size"]==2 for row in result)
    assert result[0]["response"]=="[5, 9]" and captured["do_sample"] is False


def test_batch_verification_rejects_changed_masks_and_accepts_only_identical_invalid_text():
    first=dict(response=rle("building*24"))
    same=dict(response=rle("buildings *24"))
    different=dict(response=rle("water*24"))
    assert BatchBackend.matching(first,same,(512,512))
    assert not BatchBackend.matching(first,different,(512,512))
    assert BatchBackend.matching(dict(response="water"),dict(response="water"),(512,512))
    assert not BatchBackend.matching(dict(response="water"),dict(response="class"),(512,512))


def test_invalid_answers_still_stop_after_the_recorded_budget_and_keep_raw_text(tmp_path):
    pending=jobs(tmp_path,1)
    model=FakeBatch(["water"]*3)
    with pytest.raises(ValueError,match="exhausted"):
        batch.query_tiles(model,pending)
    record=legacy.read_json(pending[0]["path"])
    assert len(model.calls)==3 and [a["response"] for a in record["attempts"]]==["water"]*3
    with pytest.raises(ValueError,match="remain invalid"):
        batch.query_tiles(FakeBatch([]),pending)


def test_v4_import_validates_provenance_redecodes_bareland_and_preserves_original(tmp_path,monkeypatch):
    source,output,data=tmp_path/"v4",tmp_path/"v5",tmp_path/"data"
    make_loveda(data)
    options=args(data,source,loveda_tile_size=512,save_images=False,format_retries=2)
    old_base=dict(protocol_version=3,adapter_source="fixture-code",checkpoint="same-checkpoint")
    model=FakeModel([rle("bareland*24")]*3)
    model.format_retries=2
    with pytest.raises(ValueError,match="bareland"):
        legacy.evaluate_dataset(options,"loveda",model,old_base)
    old_raw=next((source/"raw_text").glob("*.json"))
    original=old_raw.read_bytes()
    new_base=dict(old_base,protocol_version=4,tile_batch_size=4,alias_extension=batch.ALIAS_EXTENSION)
    options.output_dir=str(output)
    backend=FakeBatch([])
    with alias_decoder():
        receipt=import_raw_answers(source,output,options,old_base,new_base)
        expected,_,_,_=build_protocol(options,new_base)
        monkeypatch.setattr(legacy,"predict_loveda_tiled",batch.predict_tiled)
        result=legacy.evaluate_dataset(options,"loveda",backend,new_base)
    assert receipt["imported_queries"]==1 and not backend.calls
    assert result["protocol_version"]==4 and result["complete"]
    assert json.loads((output/"run_config.json").read_text())["identity"]==protocol.fingerprint(expected)
    assert old_raw.read_bytes()==original
    pred=np.asarray(Image.open(next((output/"pred_mask").glob("*.png"))))
    assert np.all(pred==4)
    changed=dict(old_base,checkpoint="another-checkpoint")
    with pytest.raises(ValueError,match="differs"):
        import_raw_answers(source,tmp_path/"bad",options,changed,new_base)


def test_output_conflict_uses_stable_separate_directory_and_preserves_existing_files(tmp_path):
    root=tmp_path/"v5_bs4"
    stored=dict(protocol_version=4,tile_batch_size=4,extension_sources_sha256={"tools/entry.py":"old"})
    requested=dict(stored,extension_sources_sha256={"tools/entry.py":"new"})
    legacy.write_json(root/"run_config.json",dict(identity=protocol.fingerprint(stored),protocol=stored))
    legacy.write_json(root/"raw_text"/"tile.json",dict(response="untouched old answer"))
    before={str(path.relative_to(root)):path.read_bytes() for path in root.rglob("*") if path.is_file()}
    actual,receipt=resolve_output_root(root,requested)
    assert actual==root.with_name(f"{root.name}_p{protocol.fingerprint(requested)[:12]}")
    assert receipt["differing_fields"]==["identity","protocol.extension_sources_sha256.tools/entry.py"]
    assert not receipt["copied_scores"] and not receipt["copied_prediction_masks"]
    assert before=={str(path.relative_to(root)):path.read_bytes() for path in root.rglob("*") if path.is_file()}
    legacy.write_json(actual/"run_config.json",dict(identity=protocol.fingerprint(requested),protocol=requested))
    assert resolve_output_root(root,requested)[0]==actual  # stable across restarts
    assert resolve_output_root(actual,requested)==(actual,None)


def test_pristine_output_is_reused_and_unidentified_artifacts_are_preserved(tmp_path):
    requested=dict(protocol_version=4,tile_batch_size=4)
    root=tmp_path/"output"
    assert resolve_output_root(root,requested)==(root,None)
    legacy.write_json(root/"raw_text"/"tile.json",dict(response="unknown protocol"))
    actual,receipt=resolve_output_root(root,requested)
    assert actual!=root and "missing" in receipt["differing_fields"][0]
    assert (root/"raw_text"/"tile.json").is_file()
    # Even a hash-named destination must match; never overwrite a collision.
    legacy.write_json(actual/"run_config.json",dict(identity="wrong",protocol=requested))
    with pytest.raises(ValueError,match="also conflicts"):
        resolve_output_root(root,requested)


def test_protocol_diff_reports_exact_fields_without_large_metadata_values():
    assert changed_fields(dict(dtype="float16",checkpoint=dict(files=["old"])),
                          dict(dtype="bfloat16",checkpoint=dict(files=["new"]))) == [
                              "protocol.checkpoint.files","protocol.dtype"]
