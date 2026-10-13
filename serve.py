"""Single-process production entry point; model inference is serialized by app.lock."""
import os
from app import app, model
from waitress import serve

if __name__ == '__main__':
    if len(os.environ.get('API_ACCESS_KEY','')) < 24:
        raise RuntimeError('Set API_ACCESS_KEY to a private access code of at least 24 characters.')
    if not model.POSE_MODEL_PATH.is_file():
        raise RuntimeError('MediaPipe model missing. Run download_model.py first.')
    serve(app, host=os.environ.get('HOST','0.0.0.0'), port=int(os.environ.get('PORT','8000')),
          threads=8, max_request_body_size=500*1024*1024)
