# Browser Version 2 validation

Tested locally on 2026-10-14 in Chromium through the real website UI.

## Formula and counter parity

`node test_browser_port.cjs` passed. All 26 dataset clips' existing Python V signals produce identical per-frame counts in the JavaScript counter. On identical real calibrated 3D input, H/V/twist and weight table match the Python reference within 1e-8. This checks the port, not fresh pose inference accuracy.

## Real video end-to-end

- S03 NEG-HOLD, height 180 cm: all 321 frames processed, 0 lifts, valid hand measurements in 100% of post-calibration frames.
- Cancel during initialization: work stopped, result values cleared, file/height/submit controls enabled again.
- Corrupt MP4: browser decoding error displayed without starting analysis.
- S01 BK-WS RF T1 PF, height 165 cm: all 298 frames processed at 30 fps, 1 lift, minimum table weight 14 kg, CSV downloaded and checked.
- Browser inference vs Python on this clip: H mean 31.7 vs 31.3 cm; mean absolute error H 0.821 cm, V 0.359 cm, twist 0.358 degrees. Weight cell agreement 92.1%. Runtime pose results are not numerically identical.
- Exported annotated MP4 decoded successfully: 1080×1920, ~30 fps, 300 frames. Skeleton, projected table and lift count visibly verified. MediaRecorder timing adds frames relative to the 298 source samples; export is not frame-exact and excludes original audio.

JavaScript syntax checks passed for model-core, analysis-worker, pose-worker and browser-app. Workers separate pose inference from OpenCV. Videos are read from browser Blob URLs; no backend upload or Python service is needed.

## Limits

This is not a validation against labelled H/V/twist ground truth. Only selected clips have been run through fresh browser inference; the 26-clip parity check reuses Python measurement signals. Motion-only counting may count bending without an object. Browser codecs and machine performance affect support and speed. Results do not persist after refresh. Variable frame rate video uses time sampling.

## Production Netlify check

GitHub main commit 4262faa automatically deployed to https://lucky-scone-171aa5.netlify.app/. The real HTTPS site loaded the local runtime assets and processed S01 at 165 cm: 298 frames, 1 lift, H mean 31.7 cm, minimum limit 14 kg. CSV downloaded from the production page and verified to contain 298 rows and final count 1. No Python backend URL or access key was configured.

Annotated video export also completed on the HTTPS site, downloaded an MP4, and restored the input controls. Proof screenshot saved locally as netlify-browser-proof.png (excluded from Git).
