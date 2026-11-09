import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import node_cache as n


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name).resolve()
        self.store=n.Store(self.base/'manager',n.Policy(node_bytes=8*1024**2,run_bytes=4*1024**2,free_bytes=0,free_fraction=0,max_chunks=2))
        self.r=self.create()

    def tearDown(self):self.tmp.cleanup()
    def create(self,request='first',budget=2*1024**2):
        return self.store.create('owner','project','task',request,self.base/'durable',budget)
    def stage(self,key='one',payload=b'a'*4096):
        src=self.base/('source-'+key);src.write_bytes(payload)
        return self.store.stage_files(self.r['id'],self.r['token'],key,{'data.bin':src})
    def get(self):return self.store.get(self.r['id'],self.r['token'])
    def copy(self):return self.store.copy_one(self.r['id'],self.r['token'])
    def finish(self,code=0):return self.store.finish(self.r['id'],self.r['token'],code)

    def test_success_reclaims_cache_keeps_exact_result_and_receipts(self):
        self.stage(payload=os.urandom(100000));self.assertTrue(self.copy())
        self.assertFalse((Path(self.r['path'])/'outbox/one').exists())
        result=self.finish();self.assertEqual(result['state'],'reclaimed')
        self.assertFalse(Path(self.r['path']).exists())
        out=Path(self.r['destination'])
        self.assertTrue(json.loads((out/'RESULT.json').read_text())['experiment_success'])
        self.assertEqual((out/'one/data.bin').read_bytes(),(self.base/'source-one').read_bytes())
        self.assertTrue(json.loads((out/'CLEANUP.json').read_text())['deleted'])
        self.assertEqual(self.store.snapshot()['reserved_bytes'],0)

    def test_failed_worker_keeps_true_exit_and_still_cleans_verified_results(self):
        self.stage();r=self.finish(7)
        self.assertEqual(r['state'],'reclaimed')
        result=n.read(Path(r['destination'])/'RESULT.json')
        self.assertEqual(result['exit']['code'],7);self.assertFalse(result['experiment_success'])

    def test_pending_count_backpressure_and_released_chunk_capacity(self):
        self.stage('one');self.stage('two')
        with self.assertRaises(n.Backpressure):self.stage('three')
        self.copy();self.stage('three')
        self.assertEqual(len(self.get()['chunks']),2)

    def test_per_run_budget_includes_scratch_and_pre_reserves_payload(self):
        (Path(self.r['path'])/'scratch/cache.bin').write_bytes(b'x'*(1900*1024))
        with self.assertRaises(n.Backpressure):self.stage(payload=b'x'*200000)
        self.assertFalse((Path(self.r['path'])/'outbox/one').exists())

    def test_stream_writer_rejects_overflow_before_bytes_are_written(self):
        folder=self.store.begin(self.r['id'],self.r['token'],'bounded',{'a.bin':4})
        with self.assertRaises(n.Backpressure):
            with self.store.output(self.r['id'],self.r['token'],'bounded','a.bin') as f:
                f.write(b'abc');f.write(b'overflow')
        self.assertEqual((Path(folder)/'a.bin').read_bytes(),b'abc')

    def test_disk_floor_blocks_new_production(self):
        with patch.object(self.store,'capacity',return_value=False):
            with self.assertRaises(n.Backpressure):self.stage()

    def test_node_budget_is_atomic_across_independent_managers(self):
        errors=[];created=[]
        def launch(i):
            try:created.append(n.Store(self.store.root,self.store.policy).create('owner','p','t','concurrent'+str(i),self.base/'other',4*1024**2))
            except n.Backpressure:errors.append(i)
        threads=[threading.Thread(target=launch,args=(i,)) for i in range(4)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(len(created),1);self.assertEqual(len(errors),3)
        self.assertLessEqual(self.store.snapshot()['reserved_bytes'],8*1024**2)

    def test_retained_failure_still_counts_against_node_budget(self):
        self.stage();out=Path(self.r['destination'])/'one';out.mkdir();(out/'data.bin').write_bytes(b'foreign')
        with self.assertRaises(ValueError):self.copy()
        self.assertTrue(Path(self.r['path']).exists());self.assertEqual(self.get()['state'],'retained')
        self.assertEqual(self.store.snapshot()['reserved_bytes'],self.r['budget'])
        self.assertEqual((out/'data.bin').read_bytes(),b'foreign')

    def test_partial_copy_resumes_only_an_exact_prefix(self):
        payload=os.urandom(50000);self.stage(payload=payload)
        out=Path(self.r['destination'])/'one';out.mkdir();(out/'data.bin').write_bytes(payload[:1234])
        self.copy();self.assertEqual((out/'data.bin').read_bytes(),payload)

    def test_copy_failure_preserves_unique_local_output(self):
        self.stage()
        with patch('node_cache.shutil.copyfileobj',side_effect=OSError('simulated NFS failure')):
            with self.assertRaises(OSError):self.copy()
        self.assertTrue((Path(self.r['path'])/'outbox/one/data.bin').exists())
        self.copy();self.assertEqual(self.finish()['state'],'reclaimed')

    def test_chunk_deletion_crash_recovers_without_double_counting(self):
        self.stage();remove=n.remove_owned_tree
        def crash(path,identity):remove(path,identity);raise OSError('crash after deletion')
        with patch('node_cache.remove_owned_tree',side_effect=crash):
            with self.assertRaises(OSError):self.copy()
        self.assertEqual(self.get()['chunks']['one']['state'],'committed')
        self.copy();self.assertEqual(self.get()['committed_bytes'],4096)
        self.assertEqual(self.finish()['state'],'reclaimed')

    def test_final_deletion_crash_is_recoverable(self):
        self.stage();self.copy();remove=n.remove_owned_tree
        def crash(path,identity):remove(path,identity);raise OSError('crash after deletion')
        with patch('node_cache.remove_owned_tree',side_effect=crash):
            with self.assertRaises(OSError):self.finish()
        self.assertEqual(self.get()['state'],'reclaiming')
        self.assertEqual(self.store.retire(self.r['id'],self.r['token'])['state'],'reclaimed')

    def test_live_or_unreadable_consumer_prevents_cleanup(self):
        with patch('node_cache.cache_consumers',return_value=[{'pid':999999,'state':'unknown'}]):
            self.assertEqual(self.finish()['state'],'retained')
        self.assertTrue(Path(self.r['path']).exists())
        self.assertEqual(self.store.retire(self.r['id'],self.r['token'])['state'],'reclaimed')

    def test_wrong_token_symlink_and_inode_replacement_are_preserved(self):
        with self.assertRaises(ValueError):self.store.finish(self.r['id'],'wrong',0)
        outside=self.base/'unrelated';outside.mkdir();(outside/'valuable').write_text('keep')
        chunk=self.store.begin(self.r['id'],self.r['token'],'bad',{'data.bin':4})
        (Path(chunk)/'data.bin').symlink_to(outside/'valuable')
        with self.assertRaises(ValueError):self.store.ready(self.r['id'],self.r['token'],'bad')
        self.assertEqual((outside/'valuable').read_text(),'keep')

    def test_idempotent_request_and_no_repeated_chunk_producer(self):
        self.assertEqual(self.create()['id'],self.r['id'])
        with self.assertRaises(ValueError):self.create(budget=1024**2)
        self.stage();self.copy()
        with self.assertRaises(ValueError):self.stage()

    def test_unfinished_building_chunk_is_retained(self):
        self.store.begin(self.r['id'],self.r['token'],'unfinished',{'data.bin':4096})
        self.assertEqual(self.finish(1)['state'],'retained')
        self.assertTrue(Path(self.r['path']).exists())

    def test_unknown_exit_never_becomes_success(self):
        r=self.finish(None);self.assertEqual(r['state'],'reclaimed')
        result=n.read(Path(r['destination'])/'RESULT.json')
        self.assertFalse(result['exit']['known']);self.assertFalse(result['experiment_success'])

    def test_supervisor_records_real_child_exit_and_watcher_cleans(self):
        result=n.supervise(self.store,self.r['id'],self.r['token'],[sys.executable,'-c','import time;time.sleep(.2);raise SystemExit(7)'])
        self.assertEqual(result['state'],'reclaimed');self.assertEqual(result['exit']['code'],7)
        self.assertTrue((Path(self.r['destination'])/'worker.log').exists())


if __name__=='__main__':unittest.main()
