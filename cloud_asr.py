"""Portable Japanese audio -> Chinese SRT entry point (no Torch required)."""
import argparse
import math
import os
from pathlib import Path
import tempfile

MODEL = '4evergr8/whisper-large-v2-translate-zh-v0.2-st-ct2'
EXTENSIONS = {'.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.opus', '.mp4', '.mkv', '.webm'}


def timestamp(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid subtitle timestamp')
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02},{ms:03}'


def collect_inputs(path):
    if path.is_file():
        files = [path] if path.suffix.lower() in EXTENSIONS else []
    elif path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        raise ValueError(f'Input does not exist: {path}')
    if not files:
        raise ValueError('No supported audio/video files found')
    names = [p.stem.casefold() for p in files]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate input stems would overwrite subtitles; process separately')
    return files


def write_srt(segments, target):
    """Publish only a complete UTF-8 file; never leave a partial final SRT."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + target.name, suffix='.tmp', dir=target.parent)
    count = 0
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as out:
            previous = 0.0
            for segment in segments:
                text = ' '.join(segment.text.split())
                if not text:
                    continue
                start, end = float(segment.start), float(segment.end)
                if not (math.isfinite(start) and math.isfinite(end)) or start < previous or end <= start:
                    raise ValueError('Invalid or nonmonotonic subtitle segment')
                previous = start
                count += 1
                out.write(f'{count}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n\n')
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Audio/video file or non-recursive directory')
    parser.add_argument('--output-dir', type=Path, default=Path('3asr'))
    parser.add_argument('--model', default=MODEL, help='Japanese-to-Chinese CT2 model ID or local directory')
    parser.add_argument('--model-dir', default='model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--compute-type', choices=['int8', 'float16', 'float32'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Require already cached/local model')
    parser.add_argument('--no-vad', action='store_true', help='Disable bundled Silero VAD')
    parser.add_argument('--dry-run', action='store_true', help='Validate paths only; no imports/downloads/inference')
    args = parser.parse_args(argv)
    try:
        files = collect_inputs(args.input)
        if args.threads < 1:
            raise ValueError('--threads must be positive')
        compute = args.compute_type or ('int8' if args.device == 'cpu' else 'float16')
        if args.device == 'cpu' and compute == 'float16':
            raise ValueError('CPU float16 unsupported; use int8 or float32')
        tasks = [(p, args.output_dir / (p.stem + '.srt')) for p in files]
        for source, target in tasks:
            if target.exists() and not args.overwrite:
                raise ValueError(f'Output exists: {target}; use --overwrite explicitly')
            if source.resolve() == target.resolve():
                raise ValueError('Output must differ from input')
        if args.dry_run:
            for source, target in tasks:
                print(f'{source} -> {target}')
            return 0
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device=args.device, compute_type=compute,
                             cpu_threads=args.threads, download_root=args.model_dir,
                             local_files_only=args.offline)
        for source, target in tasks:
            segments, _ = model.transcribe(str(source), language='ja', task='translate',
                                            vad_filter=not args.no_vad,
                                            condition_on_previous_text=False, beam_size=5)
            count = write_srt(segments, target)
            print(f'{target}: {count} subtitles' + (' (no speech detected)' if count == 0 else ''))
        return 0
    except (ValueError, ImportError, RuntimeError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
