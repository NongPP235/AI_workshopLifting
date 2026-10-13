import ast
from pathlib import Path

root = Path(__file__).parent
source = (root / 'notebook_source.py').read_text(encoding='utf-8-sig')
tree = ast.parse(source)
constants = {'MIN_CONFIDENCE','MAX_PREDICT_FRAMES','HISTORY_FRAMES','WORLD_WINDOW','SKELETON','NOSE_ABOVE_FOOT','CALIB_SECONDS','CAMERA_FOV_DEG','WAIST_FRACTION','FEET','LOWER','PNP_POINTS','TWIST_WARN_DEG','REACH_LIMITS_CM','ROW_NAMES','COL_NAMES','WEIGHT_LIMITS_KG','TABLE_LENGTH_CM','ABOVE_MARGIN_CM','HEADER_CM','PX_PER_CM','CELL_ALPHA'}
parts = ['import math\nfrom collections import deque\nimport cv2\nimport numpy as np\nimport pandas as pd\nimport mediapipe as mp\nfrom mediapipe.tasks import python as mp_python\nfrom mediapipe.tasks.python import vision as mp_vision\n']
constants.update({'LEFT_HAND', 'RIGHT_HAND'})
for item in tree.body:
    if isinstance(item, (ast.FunctionDef, ast.ClassDef)) or (isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for target in item.targets for t in ast.walk(target))):
        parts.append(ast.get_source_segment(source, item))
generated = '\n\n'.join(parts)
generated = 'from lift_counter import LiftCounter\n' + generated
generated = generated.replace('    v_history = []', '    counter = LiftCounter(info["fps"], v_threshold)\n    v_history = []')
old_start = generated.index('                    v_val = m["V"]', generated.index('def run_pipeline('))
old_end = generated.index('                    img = draw_weight_table', old_start)
generated = generated[:old_start] + '                    lift_count = counter.update(m["V"], i)\n\n' + generated[old_end:]
generated = generated.replace('                    img = draw_weight_table(img, calib, m)\n\n                img = draw_skeleton', '                    img = draw_weight_table(img, calib, m)\n                else:\n                    lift_count = counter.update(float("nan"), i)\n\n                img = draw_skeleton')
# Web adapter changes only: frame progress and validation; geometry stays unchanged.
generated = generated.replace('v_threshold=20.0):\n    cap, info = open_video(video_path)', 'v_threshold=20.0, progress=None):\n    cap, info = open_video(video_path)')
generated = generated.replace('    n_calib = int(round(CALIB_SECONDS * info["fps"]))', '    if not writer.isOpened():\n        cap.release()\n        raise ValueError("ไม่สามารถสร้างวิดีโอผลลัพธ์ได้")\n    n_calib = int(round(CALIB_SECONDS * info["fps"]))')
generated = generated.replace('            records.append(row)', '            row["lift_count_realtime"] = lift_count\n            records.append(row)')
generated = generated.replace('            if i % 50 == 0:', '            if progress and (i % 10 == 0 or i == info["frames"]):\n                progress(i, info["frames"])\n            if i % 50 == 0:')
generated = generated.replace('    calib["up"] = unit(np.mean(axes, axis=0))', '    if not axes or not np.isfinite(calib["cm_per_px"]) or calib["cm_per_px"] <= 0:\n        raise ValueError("ปรับเทียบไม่ได้ กรุณาให้เห็นทั้งตัวและยืนนิ่ง 1.5 วินาทีแรก")\n    calib["up"] = unit(np.mean(axes, axis=0))')
# Always release decoder/encoder when calibration or inference fails.
start = generated.index('def run_pipeline(')
end = generated.index('\ndef count_lifts_with_rest', start)
pipeline = generated[start:end]
block_start = pipeline.index('    with create_pose_detector() as detector:')
block_end = pipeline.index('    cap.release()\n    writer.release()', block_start)
block = pipeline[block_start:block_end]
pipeline = pipeline[:block_start] + '    try:\n' + ''.join('    '+line if line.strip() else line for line in block.splitlines(keepends=True)) + '    finally:\n        cap.release()\n        writer.release()\n' + pipeline[block_end+len('    cap.release()\n    writer.release()\n'):]
generated = generated[:start] + pipeline + generated[end:]
(root / 'model.py').write_text(generated, encoding='utf-8')
