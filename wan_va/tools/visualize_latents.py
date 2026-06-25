# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import json
import os
import sys
from pathlib import Path
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import re
import torch
from easydict import EasyDict
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import load_vae
from modules.visual_util import predictions_to_glb
from utils import init_logger, logger


DEFAULT_OBS_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
DEFAULT_VIEW_NAMES = tuple(key.rsplit(".", 1)[-1] for key in DEFAULT_OBS_KEYS)
DEFAULT_WAN22_PRETRAINED_MODEL_NAME_OR_PATH = (
    "/mi/data2T/Embodied-AI/ckpts/lingbot-vggt-base"
)
DEFAULT_VGGT_PRETRAINED_MODEL_NAME_OR_PATH = (
    "/mi/data2T/Embodied-AI/ckpts/VGGT-Omega/vggt_omega_1b_512.pt"
)
VGGT_TORCH_DTYPE = torch.float32
POINTCLOUD_BATCH_SIZE = 50


def extract_number(path):
    filename = os.path.basename(str(path))
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0


def sorted_numbered_paths(paths):
    return sorted(paths, key=extract_number)


def chunked(items, chunk_size):
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


def is_run_dir(path):
    return any(path.glob("latents_*.pt")) or any(path.glob("obs_data_*.pt"))


def find_run_dirs(root):
    root = Path(root)
    if is_run_dir(root):
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_dir() and is_run_dir(path))


def tile_camera_frames(frames):
    if not frames:
        raise ValueError("No frames to tile.")
    target_height = frames[0].shape[0]
    return np.ascontiguousarray(np.concatenate([
        resize_to_height(to_uint8_frame(frame), target_height)
        for frame in frames
    ], axis=1))


def to_uint8_frame(frame):
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)
    if frame.max(initial=0) <= 1.0 and frame.min(initial=0) >= 0.0:
        frame = frame * 255.0
    return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))


def resize_to_height(frame, target_height):
    if frame.shape[0] == target_height:
        return np.ascontiguousarray(frame)
    target_width = max(1, round(frame.shape[1] * target_height / frame.shape[0]))
    return np.asarray(Image.fromarray(frame).resize((target_width, target_height)))


def resize_to_width(frame, target_width):
    frame = to_uint8_frame(frame)
    if frame.shape[1] == target_width:
        return np.ascontiguousarray(frame)
    target_height = max(1, round(frame.shape[0] * target_width / frame.shape[1]))
    return np.asarray(Image.fromarray(frame).resize((target_width, target_height)))


def resize_to_shape(frame, target_shape):
    frame = to_uint8_frame(frame)
    target_height, target_width = target_shape
    if frame.shape[:2] == (target_height, target_width):
        return frame
    return np.asarray(Image.fromarray(frame).resize((target_width, target_height)))


def split_tshape_latent_frame(frame):
    frame = to_uint8_frame(frame)
    wrist_height = frame.shape[0] // 3
    wrist_row = frame[:wrist_height]
    high = frame[wrist_height:]
    left, right = np.split(wrist_row, 2, axis=1)
    return [high, left, right]


def stack_obs_and_latent_frames(obs_frames, latent_frames):
    num_frames = min(len(obs_frames), len(latent_frames))
    if len(obs_frames) != len(latent_frames):
        logger.warning(
            f"Frame count mismatch: obs={len(obs_frames)}, latents={len(latent_frames)}. "
            f"Using first {num_frames} frames."
        )
    combined_frames = []
    for obs_frame, latent_frame in zip(obs_frames[:num_frames], latent_frames[:num_frames]):
        latent_frame = resize_to_width(latent_frame, obs_frame.shape[1])
        combined_frames.append(np.ascontiguousarray(np.concatenate([obs_frame, latent_frame], axis=0)))
    return combined_frames


def prepare_vggt_sequence_images(view_frame_sequence):
    if not view_frame_sequence:
        raise ValueError("No frames available for point cloud reconstruction.")

    frame_batches = []
    for view_frames in view_frame_sequence:
        view_frames = [to_uint8_frame(frame) for frame in view_frames]
        target_shape = (
            max(frame.shape[0] for frame in view_frames),
            max(frame.shape[1] for frame in view_frames),
        )
        frame_batches.append(
            np.stack([resize_to_shape(frame, target_shape) for frame in view_frames], axis=0)
        )
    images = np.stack(
        frame_batches,
        axis=0,
    ).astype(np.float32) / 255.0
    return images.transpose(0, 4, 1, 2, 3)


def iter_obs_frames(run_dir, obs_keys):
    obs_paths = sorted_numbered_paths(Path(run_dir).glob("obs_data_*.pt"))
    for obs_path in obs_paths:
        obs_data = torch.load(obs_path, map_location="cpu", weights_only=False)
        entries = obs_data if isinstance(obs_data, list) else [obs_data]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            frames = [entry[key] for key in obs_keys if key in entry]
            if frames:
                yield frames


def get_obs_view_frames(run_dir, obs_keys=DEFAULT_OBS_KEYS):
    return [
        [to_uint8_frame(frame) for frame in frames]
        for frames in iter_obs_frames(run_dir, obs_keys)
    ]


def export_obs_video(run_dir, output_path=None, obs_keys=DEFAULT_OBS_KEYS, fps=10):
    run_dir = Path(run_dir)
    video_frames = get_obs_video_frames(run_dir, obs_keys)
    if not video_frames:
        logger.warning(f"No obs frames found under {run_dir}")
        return None
    if output_path is None:
        output_path = run_dir / "obs_data.mp4"
    output_path = str(output_path)
    logger.info(f"Exporting obs video with {len(video_frames)} frames to {output_path}")
    export_to_video([Image.fromarray(frame) for frame in video_frames], output_path, fps=fps)
    return output_path


def get_obs_video_frames(run_dir, obs_keys=DEFAULT_OBS_KEYS):
    return [
        tile_camera_frames(frames)
        for frames in get_obs_view_frames(run_dir, obs_keys)
    ]


def save_view_frame_images(view_frame_sequence, output_dir, view_names=DEFAULT_VIEW_NAMES):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for frame_idx, view_frames in enumerate(view_frame_sequence):
        for view_idx, frame in enumerate(view_frames):
            view_name = view_names[view_idx] if view_idx < len(view_names) else f"view_{view_idx:02d}"
            image_path = output_dir / f"frame_{frame_idx:06d}_{view_name}.png"
            Image.fromarray(to_uint8_frame(frame)).save(image_path)
            image_paths.append(image_path)
    logger.info(f"Saved {len(image_paths)} view frame images to {output_dir}")
    return image_paths


def export_combined_video(obs_frames, latent_frames, output_path, fps=10):
    combined_frames = stack_obs_and_latent_frames(obs_frames, latent_frames)
    if not combined_frames:
        logger.warning("No frames available for combined video.")
        return None
    output_path = str(output_path)
    logger.info(f"Exporting combined video with {len(combined_frames)} frames to {output_path}")
    export_to_video([Image.fromarray(frame) for frame in combined_frames], output_path, fps=fps)
    return output_path


def write_pointcloud_manifest(run_output_dir, obs_glbs, latent_glbs):
    run_output_dir = Path(run_output_dir)
    manifest_path = run_output_dir / "pointcloud_sequence_manifest.json"
    manifest = {
        "obs": [path.relative_to(run_output_dir).as_posix() for path in obs_glbs],
        "latents": [path.relative_to(run_output_dir).as_posix() for path in latent_glbs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_pointcloud_viewer(run_output_dir):
    run_output_dir = Path(run_output_dir)
    viewer_path = run_output_dir / "pointcloud_viewer.html"
    viewer_path.write_text(
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Point Cloud Sequence Viewer</title>
  <style>
    body { margin: 0; overflow: hidden; background: #111; color: #eee; font-family: sans-serif; }
    #bar { position: fixed; top: 0; left: 0; right: 0; padding: 8px; background: rgba(0,0,0,.75); z-index: 2; }
    button, select, input { margin-right: 8px; }
    canvas { display: block; width: 100vw; height: 100vh; }
  </style>
</head>
<body>
<div id="bar">
  <button id="play">Play</button>
  <button id="capture">Capture</button>
  <select id="source"><option value="obs">obs</option><option value="latents">latents</option></select>
  <input id="frame" type="range" min="0" max="0" value="0">
  <span id="label">0 / 0</span>
</div>
<canvas id="canvas"></canvas>
<script>
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl');
let manifest = {obs: [], latents: []};
const frameCache = {obs: [], latents: []};
const preloadState = {obs: {running: false, loaded: 0}, latents: {running: false, loaded: 0}};
let source = 'obs', index = 0, playing = false, pointCount = 0;
let loading = false;
let yaw = 0, pitch = -0.8, zoom = 2.6, dragging = false, lastX = 0, lastY = 0;
const frameInput = document.getElementById('frame');
const label = document.getElementById('label');
const playButton = document.getElementById('play');
const captureButton = document.getElementById('capture');
const sourceSelect = document.getElementById('source');
const PLAY_DELAY_MS = 120;
let mediaRecorder = null;
let recordedChunks = [];

function showError(error) {
  console.error(error);
  label.textContent = error?.message || String(error);
}
if (!gl) showError(new Error('WebGL is not available in this browser'));

const vertexShaderSource = `
attribute vec3 aPosition;
attribute vec3 aColor;
uniform mat4 uMvp;
varying vec3 vColor;
void main() {
  gl_Position = uMvp * vec4(aPosition, 1.0);
  gl_PointSize = 1.5;
  vColor = aColor;
}`;
const fragmentShaderSource = `
precision mediump float;
varying vec3 vColor;
void main() {
  gl_FragColor = vec4(vColor, 1.0);
}`;

function compileShader(type, sourceText) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, sourceText);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}
const program = gl.createProgram();
gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexShaderSource));
gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) showError(new Error(gl.getProgramInfoLog(program)));
gl.useProgram(program);
const positionLoc = gl.getAttribLocation(program, 'aPosition');
const colorLoc = gl.getAttribLocation(program, 'aColor');
const mvpLoc = gl.getUniformLocation(program, 'uMvp');
const positionBuffer = gl.createBuffer();
const colorBuffer = gl.createBuffer();

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0,
  ]);
}
function multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}
function translation(z) {
  return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,z,1]);
}
function rotationX(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]);
}
function rotationY(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]);
}

function updateRange() {
  const total = manifest[source]?.length || 0;
  frameInput.max = Math.max(0, total - 1);
  frameInput.value = index;
  const loaded = preloadState[source]?.loaded || 0;
  label.textContent = total ? `${index + 1} / ${total} loaded ${loaded}/${total}` : '0 / 0';
}
function componentArray(componentType, buffer, byteOffset, count) {
  if (componentType === 5126) return new Float32Array(buffer, byteOffset, count);
  if (componentType === 5121) return new Uint8Array(buffer, byteOffset, count);
  if (componentType === 5123) return new Uint16Array(buffer, byteOffset, count);
  if (componentType === 5125) return new Uint32Array(buffer, byteOffset, count);
  throw new Error(`Unsupported GLB component type ${componentType}`);
}
function accessorArray(json, bin, accessorIndex) {
  const accessor = json.accessors[accessorIndex];
  const view = json.bufferViews[accessor.bufferView];
  const itemSize = {SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4}[accessor.type];
  const byteOffset = bin.byteOffset + (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const total = accessor.count * itemSize;
  return {array: componentArray(accessor.componentType, bin.buffer, byteOffset, total), itemSize, accessor};
}
function parseGlbPointCloud(arrayBuffer) {
  const header = new DataView(arrayBuffer, 0, 12);
  if (header.getUint32(0, true) !== 0x46546c67) throw new Error('Invalid GLB file');
  let offset = 12, json = null, bin = null;
  while (offset < arrayBuffer.byteLength) {
    const view = new DataView(arrayBuffer, offset, 8);
    const length = view.getUint32(0, true);
    const type = view.getUint32(4, true);
    offset += 8;
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuffer, offset, length)));
    if (type === 0x004e4942) bin = new Uint8Array(arrayBuffer, offset, length);
    offset += length;
  }
  if (!json || !bin) throw new Error('GLB is missing JSON or BIN chunks');
  const positions = [], colors = [];
  for (const mesh of json.meshes || []) {
    for (const primitive of mesh.primitives || []) {
      if (primitive.mode !== 0 || primitive.attributes.POSITION === undefined) continue;
      const pos = accessorArray(json, bin, primitive.attributes.POSITION);
      const colorIndex = primitive.attributes.COLOR_0;
      const color = colorIndex === undefined ? null : accessorArray(json, bin, colorIndex);
      for (let i = 0; i < pos.accessor.count; i++) {
        positions.push(pos.array[i * pos.itemSize], pos.array[i * pos.itemSize + 1], pos.array[i * pos.itemSize + 2]);
        if (color) {
          const base = i * color.itemSize;
          const div = color.accessor.componentType === 5121 ? 255 : 1;
          colors.push(color.array[base] / div, color.array[base + 1] / div, color.array[base + 2] / div);
        } else {
          colors.push(1, 1, 1);
        }
      }
    }
  }
  if (!positions.length) throw new Error('No POINTS primitive found in GLB');
  const normalized = new Float32Array(positions);
  const colorData = new Float32Array(colors);
  let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < normalized.length; i += 3) {
    minX = Math.min(minX, normalized[i]); maxX = Math.max(maxX, normalized[i]);
    minY = Math.min(minY, normalized[i + 1]); maxY = Math.max(maxY, normalized[i + 1]);
    minZ = Math.min(minZ, normalized[i + 2]); maxZ = Math.max(maxZ, normalized[i + 2]);
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  const scale = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
  for (let i = 0; i < normalized.length; i += 3) {
    normalized[i] = (normalized[i] - cx) / scale * 2;
    normalized[i + 1] = (normalized[i + 1] - cy) / scale * 2;
    normalized[i + 2] = (normalized[i + 2] - cz) / scale * 2;
  }
  return {positions: normalized, colors: colorData, count: normalized.length / 3};
}
function setPointCloud(cloud) {
  pointCount = cloud.count;
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, cloud.positions, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, cloud.colors, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(colorLoc);
  gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, 0, 0);
}
async function fetchFrame(sourceName, frameIndex) {
  if (frameCache[sourceName][frameIndex]) return frameCache[sourceName][frameIndex];
  const response = await fetch(manifest[sourceName][frameIndex]);
  if (!response.ok) throw new Error(`Failed to load ${manifest[sourceName][frameIndex]}: ${response.status}`);
  const cloud = parseGlbPointCloud(await response.arrayBuffer());
  frameCache[sourceName][frameIndex] = cloud;
  preloadState[sourceName].loaded = frameCache[sourceName].filter(Boolean).length;
  if (sourceName === source) updateRange();
  return cloud;
}
async function preloadSource(sourceName) {
  if (preloadState[sourceName].running) return;
  preloadState[sourceName].running = true;
  try {
    const total = manifest[sourceName]?.length || 0;
    for (let frameIndex = 0; frameIndex < total; frameIndex++) {
      await fetchFrame(sourceName, frameIndex);
    }
  } catch (error) {
    showError(error);
  } finally {
    preloadState[sourceName].running = false;
  }
}
async function loadFrame(i) {
  if (loading) return false;
  loading = true;
  try {
    const total = manifest[source]?.length || 0;
    if (!total) { updateRange(); return false; }
    const nextIndex = Math.max(0, Math.min(i, total - 1));
    const cloud = frameCache[source][nextIndex] || await fetchFrame(source, nextIndex);
    setPointCloud(cloud);
    index = nextIndex;
    updateRange();
    return true;
  } catch (error) {
    showError(error);
    return false;
  } finally {
    loading = false;
  }
}
function resize() {
  canvas.width = innerWidth * devicePixelRatio;
  canvas.height = innerHeight * devicePixelRatio;
  gl.viewport(0, 0, canvas.width, canvas.height);
}
function render() {
  resize();
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.07, 0.07, 0.07, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const proj = perspective(Math.PI / 4, canvas.width / canvas.height, 0.01, 100);
  const model = multiply(translation(-zoom), multiply(rotationY(yaw), rotationX(pitch)));
  gl.uniformMatrix4fv(mvpLoc, false, multiply(proj, model));
  if (pointCount) gl.drawArrays(gl.POINTS, 0, pointCount);
  requestAnimationFrame(render);
}
playButton.onclick = () => {
  playing = !playing;
  playButton.textContent = playing ? 'Pause' : 'Play';
  if (playing) playNextFrame();
};
function recordingMimeType() {
  const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  return types.find(type => MediaRecorder.isTypeSupported(type)) || '';
}
function recordingFileName() {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  return `pointcloud_${source}_${stamp}.webm`;
}
function downloadRecording(blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = recordingFileName();
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function startCapture() {
  if (!canvas.captureStream || typeof MediaRecorder === 'undefined') {
    showError(new Error('Canvas recording is not supported in this browser'));
    return;
  }
  recordedChunks = [];
  const stream = canvas.captureStream(30);
  const mimeType = recordingMimeType();
  mediaRecorder = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);
  mediaRecorder.ondataavailable = event => {
    if (event.data && event.data.size > 0) recordedChunks.push(event.data);
  };
  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, {type: mediaRecorder.mimeType || 'video/webm'});
    recordedChunks = [];
    mediaRecorder = null;
    captureButton.textContent = 'Capture';
    if (blob.size > 0) downloadRecording(blob);
  };
  mediaRecorder.start();
  captureButton.textContent = 'Stop';
}
function stopCapture() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
}
captureButton.onclick = () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    stopCapture();
  } else {
    startCapture();
  }
};
sourceSelect.onchange = async e => {
  playing = false;
  playButton.textContent = 'Play';
  source = e.target.value;
  preloadSource(source);
  await loadFrame(0);
};
frameInput.oninput = async e => {
  playing = false;
  playButton.textContent = 'Play';
  await loadFrame(Number(e.target.value));
};
canvas.onmousedown = e => { dragging = true; lastX = e.clientX; lastY = e.clientY; };
canvas.onmouseup = () => { dragging = false; };
canvas.onmouseleave = () => { dragging = false; };
canvas.onmousemove = e => {
  if (!dragging) return;
  yaw += (e.clientX - lastX) * 0.01;
  pitch += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
};
canvas.onwheel = e => {
  e.preventDefault();
  zoom = Math.max(0.4, Math.min(20, zoom * Math.exp(e.deltaY * 0.001)));
};
async function playNextFrame() {
  const total = manifest[source]?.length || 0;
  if (!playing || !total) return;
  await loadFrame((index + 1) % total);
  if (playing) setTimeout(playNextFrame, PLAY_DELAY_MS);
}
fetch('pointcloud_sequence_manifest.json')
  .then(r => {
    if (!r.ok) throw new Error(`Failed to load manifest: ${r.status}`);
    return r.json();
  })
  .then(data => {
    manifest = data;
    updateRange();
    preloadSource(source);
    preloadSource(source === 'obs' ? 'latents' : 'obs');
    return loadFrame(0);
  })
  .catch(showError);
render();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    write_pointcloud_viewer_server(run_output_dir)
    return viewer_path


def write_pointcloud_viewer_server(run_output_dir):
    server_path = Path(run_output_dir) / "pointcloud_viewer_server.py"
    server_path.write_text(
        '''#!/usr/bin/env python3
import argparse
import functools
import http.server
import socketserver
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    class ViewerRequestHandler(http.server.SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def end_headers(self):
            self.send_header("Connection", "close")
            super().end_headers()

    handler = functools.partial(ViewerRequestHandler, directory=str(root))

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
        display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        print(f"Serving {root}")
        print(f"Open http://{display_host}:{args.port}/pointcloud_viewer.html")
        print("If this is a remote server, forward the port, for example:")
        print(f"  ssh -L {args.port}:127.0.0.1:{args.port} <user>@<server>")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )
    server_path.chmod(0o755)
    return server_path


def build_vggt_adapter(args):
    from modules.vggt_adapter import VGGTAdapter

    device = torch.device(args.device)
    adapter_config = {
        "default_image_size": [512, 512],
        "latent_frame_mode": "concat",
        "latent_dimension": 2048,
    }
    adapter = VGGTAdapter.from_pretrained(
        vggt_adapter_config=adapter_config,
        vggt_pretrained_path=args.vggt_pretrained_model_name_or_path,
        device=device,
        torch_dtype=VGGT_TORCH_DTYPE,
    )
    adapter.eval()
    return adapter


def pointcloud_scene_from_predictions(predictions, target_dir):
    return predictions_to_glb(
        predictions,
        conf_thres=40.0,
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=True,
        mask_sky=False,
        target_dir=str(target_dir),
        max_points=1000000
    )


def save_pointcloud_frame(predictions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = pointcloud_scene_from_predictions(predictions, output_path.parent)
    scene.export(output_path)
    return output_path


def split_prediction_sequence(predictions, expected_frame_count, squeeze_batch_dim=False):
    split_predictions = []
    for frame_idx in range(expected_frame_count):
        frame_predictions = {}
        for key, value in predictions.items():
            if isinstance(value, np.ndarray) and value.shape[:1] == (expected_frame_count,):
                if squeeze_batch_dim:
                    frame_predictions[key] = value[frame_idx]
                else:
                    frame_predictions[key] = value[frame_idx: frame_idx + 1]
            else:
                frame_predictions[key] = value
        split_predictions.append(frame_predictions)
    return split_predictions


def save_pointcloud_sequence(
    adapter,
    view_frame_sequence,
    output_dir,
    device,
    pointcloud_batch_size=POINTCLOUD_BATCH_SIZE,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_paths = []
    logger.info(
        f"Reconstructing point cloud sequence with {len(view_frame_sequence)} frames "
        f"-> {output_dir.name}"
    )
    if not view_frame_sequence:
        logger.warning(f"No frames available for point cloud reconstruction under {output_dir}")
        return glb_paths
    if pointcloud_batch_size < 1:
        raise ValueError("pointcloud_batch_size must be >= 1")

    for frame_offset in range(0, len(view_frame_sequence), pointcloud_batch_size):
        frame_batch = view_frame_sequence[frame_offset: frame_offset + pointcloud_batch_size]
        logger.info(
            f"Reconstructing frames {frame_offset}..{frame_offset + len(frame_batch) - 1} "
            f"with batch size {len(frame_batch)}"
        )
        images = prepare_vggt_sequence_images(frame_batch)
        with torch.no_grad():
            predictions = adapter(
                images,
                return_latents=False,
                device=device,
                torch_dtype=VGGT_TORCH_DTYPE,
            )
        for batch_frame_idx, frame_predictions in enumerate(
            split_prediction_sequence(predictions, len(frame_batch), squeeze_batch_dim=True)
        ):
            frame_idx = frame_offset + batch_frame_idx
            glb_paths.append(save_pointcloud_frame(frame_predictions, output_dir / f"frame_{frame_idx:06d}.glb"))
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
        torch.cuda.empty_cache()

    return glb_paths


def save_pointcloud_outputs(
    adapter,
    obs_view_frames,
    latent_view_frames,
    run_output_dir,
    device,
    pointcloud_batch_size,
):
    output_dir = Path(run_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_glbs = save_pointcloud_sequence(
        adapter,
        obs_view_frames,
        output_dir / "obs_pointcloud_frames",
        device,
        pointcloud_batch_size,
    )
    latent_glbs = save_pointcloud_sequence(
        adapter,
        latent_view_frames,
        output_dir / "latents_pointcloud_frames",
        device,
        pointcloud_batch_size,
    )
    manifest_path = write_pointcloud_manifest(output_dir, obs_glbs, latent_glbs)
    write_pointcloud_viewer(output_dir)
    return manifest_path


def patch_torch_npu_if_needed(device):
    if device == "cpu":
        return
    import torch_npu
    from torch_npu.contrib import transfer_to_npu


class VA_Server:

    def __init__(self, job_config):
        self.job_config = job_config
        self.device = torch.device(
            getattr(job_config, "visualize_device", f"cuda:{job_config.local_rank}")
        )
        self.dtype = torch.float32 if self.device.type == "cpu" else job_config.param_dtype

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )

        self.video_processor = VideoProcessor(vae_scale_factor=1)

    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video.detach(), output_type=output_type)
        return video

    def decode_video_latent(
        self,
        latent_root,
        chunk_size=5,
    ):
        pt_pathes = sorted_numbered_paths(Path(latent_root).glob("latents_*.pt"))
        if not pt_pathes:
            logger.warning(f"No latents_*.pt files found under {latent_root}")
            return None
        chunked_pathes = chunked(pt_pathes, chunk_size)

        # Collect all video frames from all chunks
        all_frames = []
        for index, chunk in enumerate(chunked_pathes):
            logger.info(f"Decoding chunk {index + 1}/{len(chunked_pathes)}")
            pred_latent_lst = []
            for pt_path in chunk:
                pred_latent_lst.append(torch.load(pt_path, weights_only=False).to(self.device))
            pred_latent_lst = torch.cat(pred_latent_lst, dim=2)
            decoded_video = self.decode_one_video(pred_latent_lst, 'np')[0]
            all_frames.append(decoded_video)
            del pred_latent_lst, decoded_video
            if hasattr(torch, "npu"):
                torch.npu.empty_cache()
            torch.cuda.empty_cache()

        all_frames = np.concatenate(all_frames, axis=0)
        latent_view_frames = [
            split_tshape_latent_frame(frame)
            for frame in all_frames
        ]
        latent_video_frames = [
            tile_camera_frames(frames)
            for frames in latent_view_frames
        ]
        return latent_video_frames, latent_view_frames


def build_model(args):
    patch_torch_npu_if_needed(args.device)
    config = EasyDict()
    config.wan22_pretrained_model_name_or_path = args.wan22_pretrained_model_name_or_path
    config.param_dtype = torch.bfloat16
    config.rank = int(os.getenv("RANK", 0))
    config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.latent_root = str(args.latent_root)
    config.visualize_device = args.device
    return VA_Server(config)


def run(args):
    run_dirs = find_run_dirs(args.latent_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {args.latent_root}")
    if not args.save_video and not args.save_pointcloud and not args.save_frame_images:
        logger.warning("save_video, save_pointcloud, and save_frame_images are disabled; nothing to do.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    for run_dir in run_dirs:
        logger.info(f"Visualizing run: {run_dir}")
        run_output_dir = Path(run_dir)
        if output_dir is not None:
            run_output_dir = output_dir / Path(run_dir).name
            run_output_dir.mkdir(parents=True, exist_ok=True)
        obs_view_frames = get_obs_view_frames(run_dir, DEFAULT_OBS_KEYS)
        obs_frames = [
            tile_camera_frames(frames)
            for frames in obs_view_frames
        ]
        latent_frames = latent_view_frames = None
        if args.save_video or args.save_pointcloud or args.save_frame_images:
            model = build_model(args)
            decoded_latents = model.decode_video_latent(
                str(run_dir),
                chunk_size=5,
            )
            if decoded_latents is None:
                latent_frames, latent_view_frames = [], []
            else:
                latent_frames, latent_view_frames = decoded_latents
            del model
            if hasattr(torch, "npu"):
                torch.npu.empty_cache()
            torch.cuda.empty_cache()

        if args.save_frame_images:
            save_view_frame_images(obs_view_frames, run_output_dir / "obs_frames")
            if latent_view_frames is not None:
                save_view_frame_images(latent_view_frames, run_output_dir / "latents_frames")
            else:
                logger.warning("No latent frames available for frame image export.")

        if args.save_video:
            export_combined_video(
                obs_frames,
                latent_frames,
                run_output_dir / "obs_latents_compare.mp4",
                fps=10,
            )
        if not args.save_pointcloud:
            continue

        vggt_device = torch.device(args.device)
        vggt_adapter = build_vggt_adapter(args)
        save_pointcloud_outputs(
            vggt_adapter,
            obs_view_frames,
            latent_view_frames,
            run_output_dir,
            vggt_device,
            args.pointcloud_batch_size,
        )
        del vggt_adapter
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
        torch.cuda.empty_cache()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=None,
        help=f"Root containing run directories."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for generated videos. Defaults to writing inside each run directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="npu:15",
        help=f"Decode device for latents."
    )
    parser.add_argument(
        "--wan22-pretrained-model-name-or-path",
        type=str,
        default=DEFAULT_WAN22_PRETRAINED_MODEL_NAME_OR_PATH,
        help="Path to the Wan2.2 pretrained model directory."
    )
    parser.add_argument(
        "--vggt-pretrained-model-name-or-path",
        type=str,
        default=DEFAULT_VGGT_PRETRAINED_MODEL_NAME_OR_PATH,
        help="Path to the VGGT-Omega checkpoint."
    )
    parser.add_argument(
        "--pointcloud-batch-size",
        type=int,
        default=POINTCLOUD_BATCH_SIZE,
        help=(
            "Batch size for the multiview_frame_batch point cloud reconstruction."
        )
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate obs/latents comparison MP4."
    )
    parser.add_argument(
        "--save-pointcloud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate per-frame VGGT-Omega point cloud GLBs and viewer."
    )
    parser.add_argument(
        "--save-frame-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-frame, per-view obs and latent images as PNG files."
    )
    return parser


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    # setup_debugger()
    init_logger()
    main()
