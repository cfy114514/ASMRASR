import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch, MagicMock
import cloud_asr as app

class CloudTests(unittest.TestCase):
    def test_timestamp(self):
        self.assertEqual(app.timestamp(3661.234), '01:01:01,234')
        self.assertEqual(app.timestamp(59.9999), '00:01:00,000')
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): app.timestamp(value)

    def test_formats_and_collision(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('日语.FLAC','b.aac','c.m4a','ignore.txt'): (root/name).touch()
            self.assertEqual(len(app.collect_inputs(root)),3)
            (root/'b.wav').touch()
            with self.assertRaises(ValueError): app.collect_inputs(root)

    def test_missing_empty(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): app.collect_inputs(Path(d))
            with self.assertRaises(ValueError): app.collect_inputs(Path(d)/'missing')

    def test_atomic_srt_and_failure(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'中文.srt'
            self.assertEqual(app.write_srt([S(start=0,end=1.2,text=' 中文\n字幕 '),S(start=2,end=3,text=' ')],target),1)
            self.assertEqual(target.read_text(), '1\n00:00:00,000 --> 00:00:01,200\n中文字幕\n\n'.replace('中文字幕','中文 字幕'))
            old=target.read_bytes()
            def broken():
                yield S(start=0,end=1,text='test')
                raise RuntimeError('inference interrupted')
            with self.assertRaises(RuntimeError): app.write_srt(broken(),target)
            self.assertEqual(target.read_bytes(),old)
            self.assertEqual(list(Path(d).glob('*.tmp')),[])

    def test_invalid_segments(self):
        with tempfile.TemporaryDirectory() as d:
            for start,end in ((-1,1),(2,1),(0,float('nan'))):
                with self.assertRaises(ValueError): app.write_srt([S(start=start,end=end,text='x')],Path(d)/'x.srt')

    def test_dry_run_no_download(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.wav'; p.touch()
            self.assertEqual(app.main([str(p),'--dry-run']),0)
            out=Path(d)/'output'; out.mkdir(); (out/'input.srt').touch()
            with self.assertRaises(SystemExit): app.main([str(p),'--output-dir',str(out),'--dry-run'])

    def test_inference_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.wav'; p.touch()
            module=MagicMock(); module.WhisperModel.return_value.transcribe.return_value=(iter([S(start=0,end=1,text='你好')]),None)
            with patch.dict('sys.modules', {'faster_whisper':module}):
                self.assertEqual(app.main([str(p),'--output-dir',d,'--offline']),0)
            self.assertEqual(module.WhisperModel.call_args.kwargs['compute_type'],'int8')
            self.assertTrue(module.WhisperModel.call_args.kwargs['local_files_only'])
            self.assertEqual(module.WhisperModel.return_value.transcribe.call_args.kwargs['language'],'ja')
            self.assertIn('你好',(Path(d)/'input.srt').read_text())

if __name__ == '__main__': unittest.main()
