from pathlib import Path

path = Path('/opt/conda/lib/python3.11/site-packages/ivrit/audio.py')
text = path.read_text()
old = """            segments, info = self.model_object.transcribe(audio_path, language=language, word_timestamps=output_options['word_timestamps'])"""
new = """            # ivrit 0.2.6 drops newer faster-whisper kwargs. Forward the
            # language-routing and transcription-task controls explicitly.
            whisper_options = {
                'task': kwargs.get('task', 'transcribe'),
                'multilingual': kwargs.get('multilingual', False),
                'language_detection_threshold': kwargs.get('language_detection_threshold', 0.5),
                'language_detection_segments': kwargs.get('language_detection_segments', 1),
            }
            segments, info = self.model_object.transcribe(
                audio_path,
                language=language,
                word_timestamps=output_options['word_timestamps'],
                **whisper_options,
            )"""
if old not in text:
    raise SystemExit('ivrit transcribe call was not found; refusing an unsafe patch')
path.write_text(text.replace(old, new, 1))
print('patched', path)
