from pathlib import Path
import os
import urllib.request

path=Path(os.environ.get('POSE_MODEL_PATH',str(Path(__file__).parent/'models'/'pose_landmarker_lite.task')))
path.parent.mkdir(parents=True,exist_ok=True)
if not path.is_file():
    temp=path.with_suffix('.download')
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',temp)
    temp.replace(path)
print('MediaPipe model ready')
