# Hand-Controlled 3D Particle System

Real-time interactive Three.js particle system driven by **hand gestures** via the camera (MediaPipe Hand Landmarker). You can expand/contract the cloud, shift colors, and switch between five particle templates: **Hearts**, **Flowers**, **Saturn**, **Fireworks**, and **Stars**.

## How to run

Camera access requires a **secure context** (HTTPS or `localhost`). Serve the project over HTTP:

```bash
# Node
npx serve .

# Python 3
python -m http.server 8080
```

Then open **http://localhost:3000** (or 8080) in Chrome.

## Gesture controls

| Gesture | Effect |
|--------|--------|
| **Open palm** | Expand particles |
| **Fist / closed hand** | Contract particles |
| **1–5 fingers extended** | Switch shape: 1→Hearts, 2→Flowers, 3→Saturn, 4→Fireworks, 5→Stars |
| **Hand up/down** | Shift color (hue) |

## Keyboard fallback

When the camera isn’t available or hand tracking hasn’t loaded, you can still control the scene:

- **1–5** – Select template (Hearts, Flowers, Saturn, Fireworks, Stars)
- **↑ / ↓** – Expand / contract
- **Q / W** – Decrease / increase color hue

## Tech
- **Three.js** – 3D scene and point particle system
- **@mediapipe/tasks-vision** – Hand landmark detection (script tag)
- No build step; open `index.html` via a local server for camera support.

................................................................................................................................................................................................................................................................................
