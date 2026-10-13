"""Download and validate native dependencies/model during a non-Docker build."""
import download_model
import app
import imageio_ffmpeg
import subprocess

with app.model.create_pose_detector() as detector:
    pass
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-version'],check=True,capture_output=True)
print('Native backend ready: MediaPipe, OpenCV and FFmpeg verified')
