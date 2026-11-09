import unittest,tempfile,os
from pathlib import Path
import node_cache as n
class SymlinkAccounting(unittest.TestCase):
 def test_link_targets_are_not_traversed_or_charged_twice(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cache=root/'cache';cache.mkdir();outside=root/'outside';outside.mkdir();(outside/'huge').write_bytes(b'x'*1048576)
   (cache/'file').write_bytes(b'x'*4096);(cache/'file_link').symlink_to(outside/'huge');(cache/'directory_link').symlink_to(outside,target_is_directory=True)
   expected=sum(p.lstat().st_blocks*512 for p in cache.iterdir())
   self.assertEqual(n.allocated(cache),expected)
 def test_existing_runtime_links_do_not_bypass_logical_budget(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);store=n.Store(root/'manager',n.Policy(node_bytes=8*1024**2,run_bytes=4*1024**2,free_bytes=0,free_fraction=0,max_chunks=2))
   a=store.create('owner','project','task','a',root/'out',2*1024**2)
   (Path(a['path'])/'scratch/python').symlink_to('/usr/bin/python3')
   b=store.create('owner','project','task','b',root/'out',2*1024**2)
   with self.assertRaises(n.Backpressure):store.create('owner','project','task','c',root/'out',2*1024**2)
