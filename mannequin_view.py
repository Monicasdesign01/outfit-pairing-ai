"""
Step 9 (stretch) - a 3D preview showing the matched outfit actually worn.

Rewritten twice on 2026-09-06: the first version used solid-colour
capsules only (didn't look like "wearing clothes"); this version textures
the real garment cutout photos onto flat panels held in front of a
mannequin body, styled deliberately like a wooden artist's posable
mannequin (a neutral, jointed figure - not an attempt at photorealistic
skin) rather than a "balloon" made of a few oversized shapes. A second
pass added: a studio backdrop and floor shadow so it doesn't look like a
bare test scene, and a camera that measures the actual figure and frames
all of it (head to feet) automatically, so nothing gets cropped
regardless of proportions.

Still deliberately not a downloaded 3D model file - the body is built
from basic Three.js shapes so this costs zero extra weight against the
tight Streamlit Cloud memory budget (Section 7). The garment textures are
the real photos already shown elsewhere on the page - nothing new is
fetched or computed to make this work.
"""

import base64

from color_swatches import to_hex
from pairing_rules import TOPS, BOTTOMS

_THREE_JS_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"

# A neutral, slightly warm tone for the mannequin body itself - matching
# how a real wooden/plastic artist's mannequin looks (one consistent
# material, not skin-toned), used everywhere a garment image doesn't
# cover the figure.
MANNEQUIN_BASE_COLOR = "dcd3c3"


def image_path_to_data_uri(path):
    with open(path, "rb") as f:
        return image_bytes_to_data_uri(f.read())


def image_bytes_to_data_uri(image_bytes):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def resolve_body_visuals(category_a, color_a, image_a, category_b, color_b, image_b):
    """
    Works out which item (and its cutout image, if there is one) goes on
    the torso/arms vs the legs - whichever order the uploaded item and
    the chosen match happen to come in, since either one could be the top
    or the bottom. A dress covers the whole figure. Returns a dict with
    "torso_image"/"legs_image" (a data URI, or None to fall back to the
    plain mannequin colour) and "torso_color"/"legs_color" (used for that
    fallback).
    """
    visuals = {"torso_color": "gray", "legs_color": "gray", "torso_image": None, "legs_image": None}

    for category, color, image in ((category_a, color_a, image_a), (category_b, color_b, image_b)):
        if category == "dress":
            visuals["torso_color"] = visuals["legs_color"] = color
            visuals["torso_image"] = visuals["legs_image"] = image
        elif category in TOPS or category == "blazer":
            visuals["torso_color"] = color
            visuals["torso_image"] = image
        elif category in BOTTOMS:
            visuals["legs_color"] = color
            visuals["legs_image"] = image

    return visuals


def render_mannequin_html(visuals, height=480):
    torso_hex = to_hex(visuals["torso_color"])
    legs_hex = to_hex(visuals["legs_color"])
    torso_image = visuals["torso_image"] or ""
    legs_image = visuals["legs_image"] or ""

    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  html, body {{ margin: 0; overflow: hidden; }}
  body {{
    background: radial-gradient(ellipse at center, #fafafa 0%, #e9e9ec 100%);
  }}
  #label {{
    position: absolute; bottom: 6px; width: 100%; text-align: center;
    font-family: sans-serif; font-size: 12px; color: #999;
  }}
</style>
</head>
<body>
<div id="label">Drag to rotate</div>
<script type="module">
import * as THREE from "{_THREE_JS_CDN}";

const container = document.body;
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(32, window.innerWidth / {height}, 0.1, 100);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setSize(window.innerWidth, {height});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

// --- Studio backdrop: a soft vertical-gradient "paper" sweep behind the
// figure, drawn once on a canvas rather than loading an image file.
function makeGradientTexture(topColor, bottomColor) {{
    const c = document.createElement("canvas");
    c.width = 4; c.height = 256;
    const ctx = c.getContext("2d");
    const grad = ctx.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, topColor);
    grad.addColorStop(1, bottomColor);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 4, 256);
    return new THREE.CanvasTexture(c);
}}
const backdrop = new THREE.Mesh(
    new THREE.PlaneGeometry(10, 7),
    new THREE.MeshBasicMaterial({{ map: makeGradientTexture("#f6f6f4", "#dcdcdf"), depthWrite: false }})
);
backdrop.position.set(0, 1.6, -2.2);
scene.add(backdrop);

// Soft round "contact shadow" under the feet - a cheap, reliable stand-in
// for real shadow mapping (which needs careful tuning to avoid artifacts
// at this small a scale) using a radial-gradient texture on a flat disc.
function makeShadowTexture() {{
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    const ctx = c.getContext("2d");
    const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, "rgba(0,0,0,0.32)");
    grad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
}}
const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(1.5, 0.6),
    new THREE.MeshBasicMaterial({{ map: makeShadowTexture(), transparent: true, depthWrite: false }})
);
shadow.rotation.x = -Math.PI / 2;
shadow.position.set(0, 0.001, 0.1);
scene.add(shadow);

// --- Lighting: a simple three-point setup instead of one flat light, so
// the figure reads as a solid 3D object rather than a flat cutout.
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
keyLight.position.set(2.2, 3.2, 3);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.35);
fillLight.position.set(-2.5, 1.5, 2);
scene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0xffffff, 0.4);
rimLight.position.set(0, 2.5, -3);
scene.add(rimLight);

// --- The mannequin body, built foot-up so every joint sits flush
// against the one below it (each segment's half-height already
// includes its own capsule end-caps, so no gaps or overlaps).
const baseMat = new THREE.MeshStandardMaterial({{ color: 0x{MANNEQUIN_BASE_COLOR}, roughness: 0.6 }});
const torsoImage = "{torso_image}";
const legsImage = "{legs_image}";
const torsoColor = 0x{torso_hex[1:]};
const legsColor = 0x{legs_hex[1:]};
const torsoMat = new THREE.MeshStandardMaterial({{ color: torsoImage ? 0x{MANNEQUIN_BASE_COLOR} : torsoColor, roughness: 0.6 }});
const legsMat = new THREE.MeshStandardMaterial({{ color: legsImage ? 0x{MANNEQUIN_BASE_COLOR} : legsColor, roughness: 0.6 }});

const figure = new THREE.Group();

function capsule(radius, length, material) {{
    return new THREE.Mesh(new THREE.CapsuleGeometry(radius, length, 8, 16), material);
}}

// Feet
const footGeo = new THREE.BoxGeometry(0.16, 0.09, 0.32);
let y = 0.045;
for (const side of [-1, 1]) {{
    const foot = new THREE.Mesh(footGeo, baseMat);
    foot.position.set(side * 0.13, y, 0.06);
    figure.add(foot);
}}
y = 0.09;

// Shins
const shinLen = 0.55, shinR = 0.1;
const shinCenterY = y + shinLen / 2 + shinR - 0.03;
for (const side of [-1, 1]) {{
    const shin = capsule(shinR, shinLen, legsMat);
    shin.position.set(side * 0.15, shinCenterY, 0);
    figure.add(shin);
}}
y = shinCenterY + shinLen / 2 + shinR;

// Thighs
const thighLen = 0.5, thighR = 0.14;
const thighCenterY = y + thighLen / 2 + thighR - 0.05;
for (const side of [-1, 1]) {{
    const thigh = capsule(thighR, thighLen, legsMat);
    thigh.position.set(side * 0.15, thighCenterY, 0);
    figure.add(thigh);
}}
y = thighCenterY + thighLen / 2 + thighR;

// Hips/waist - a single piece bridging the legs and torso
const hipLen = 0.14, hipR = 0.25;
const hipCenterY = y + hipLen / 2 + hipR - 0.08;
const hip = capsule(hipR, hipLen, legsMat);
hip.position.set(0, hipCenterY, 0);
figure.add(hip);
y = hipCenterY + hipLen / 2 + hipR;

// Torso
const torsoLen = 0.62, torsoR = 0.3;
const torsoCenterY = y + torsoLen / 2 + torsoR - 0.12;
const torsoBottomY = torsoCenterY - torsoLen / 2 - torsoR;  // where a bottom's waistband should reach up to
const torso = capsule(torsoR, torsoLen, torsoMat);
torso.position.set(0, torsoCenterY, 0);
figure.add(torso);
y = torsoCenterY + torsoLen / 2 + torsoR;

// Shoulders (a touch wider than the torso, reads more human than a
// straight cylinder going directly into arms)
const shoulderY = y - 0.16;

// Arms - upper arm + forearm, hanging naturally at a slight outward angle
const upperArmLen = 0.4, upperArmR = 0.085;
const foreArmLen = 0.38, foreArmR = 0.07;
for (const side of [-1, 1]) {{
    const upperArm = capsule(upperArmR, upperArmLen, torsoMat);
    upperArm.position.set(side * 0.36, shoulderY - 0.18, 0);
    upperArm.rotation.z = side * 0.22;
    figure.add(upperArm);

    const foreArm = capsule(foreArmR, foreArmLen, baseMat);
    foreArm.position.set(side * 0.47, shoulderY - 0.58, 0.05);
    foreArm.rotation.z = side * 0.1;
    foreArm.rotation.x = 0.15;
    figure.add(foreArm);

    const hand = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 12), baseMat);
    hand.position.set(side * 0.5, shoulderY - 0.82, 0.1);
    figure.add(hand);
}}

// Neck
const neckLen = 0.1, neckR = 0.075;
const neckCenterY = y + neckLen / 2 + neckR - 0.05;
const neck = capsule(neckR, neckLen, baseMat);
neck.position.set(0, neckCenterY, 0);
figure.add(neck);
y = neckCenterY + neckLen / 2 + neckR;

// Head
const headR = 0.22;
const head = new THREE.Mesh(new THREE.SphereGeometry(headR, 24, 24), baseMat);
head.position.set(0, y + headR - 0.05, 0);
figure.add(head);

const neckBaseY = neckCenterY - neckLen / 2 - neckR;   // where a top's collar should stop
const hipBaseY = hipCenterY - hipLen / 2 - hipR;        // where a bottom's waistband starts
const shoulderWidth = 0.95;

figure.position.y = -1.1;  // recenters the whole figure closer to the scene origin
scene.add(figure);

// --- Garment panels: the real cutout photo, held in front of the body,
// sized to its own aspect ratio and capped so a top doesn't creep over
// the head or a bottom past the ankles.
function addGarmentPanel(dataUri, topY, bottomY, zOffset) {{
    if (!dataUri) return;
    new THREE.TextureLoader().load(dataUri, (texture) => {{
        texture.colorSpace = THREE.SRGBColorSpace;
        const aspect = texture.image.width / texture.image.height;
        const maxHeight = topY - bottomY;
        const maxWidth = shoulderWidth * 1.5;
        let panelHeight = maxHeight;
        let panelWidth = panelHeight * aspect;
        if (panelWidth > maxWidth) {{
            panelWidth = maxWidth;
            panelHeight = panelWidth / aspect;
        }}
        const panel = new THREE.Mesh(
            new THREE.PlaneGeometry(panelWidth, panelHeight),
            new THREE.MeshStandardMaterial({{
                map: texture, transparent: true, alphaTest: 0.3,
                side: THREE.DoubleSide, roughness: 0.8,
            }})
        );
        // Anchored from the top (shoulder for a shirt, waist for pants) -
        // that's the point a garment actually hangs from on a body, so any
        // shortfall from the aspect-ratio fit shows up as a shorter hem at
        // the bottom instead of a gap at the shoulder or waist, which
        // would be far more noticeable.
        panel.position.set(0, figure.position.y + topY - panelHeight / 2, zOffset);
        scene.add(panel);
        fitCameraToFigure();
    }});
}}

// The two panels meet right where the torso becomes the hip, each
// anchored from its own natural hang-point (shoulder / waist) - a small
// depth offset between them avoids z-fighting where their edges meet.
addGarmentPanel(torsoImage, neckBaseY, torsoBottomY - 0.05, 0.36);
addGarmentPanel(legsImage, torsoBottomY, 0.05, 0.4);

// --- Camera: measure the actual figure (plus anything added to the
// scene, like garment panels) and back off exactly far enough that all
// of it - head to feet, full width - fits in frame. Re-run whenever a
// garment image finishes loading, since that can change the bounds.
function fitCameraToFigure() {{
    const box = new THREE.Box3();
    box.expandByObject(figure);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    const vFov = camera.fov * Math.PI / 180;
    const distForHeight = (size.y / 2) / Math.tan(vFov / 2);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
    const distForWidth = (size.x / 2) / Math.tan(hFov / 2);
    const distance = Math.max(distForHeight, distForWidth) * 1.28;

    camera.position.set(center.x, center.y, center.z + distance);
    camera.lookAt(center);
}}
fitCameraToFigure();

let dragging = false, lastX = 0;
renderer.domElement.addEventListener("pointerdown", (e) => {{ dragging = true; lastX = e.clientX; }});
window.addEventListener("pointerup", () => dragging = false);
window.addEventListener("pointermove", (e) => {{
    if (dragging) {{
        figure.rotation.y += (e.clientX - lastX) * 0.01;
        lastX = e.clientX;
    }}
}});

// Gentle turntable oscillation rather than a full spin - the garment
// panels are flat images and would vanish edge-on at 90 degrees.
let angle = 0;
function animate() {{
    requestAnimationFrame(animate);
    if (!dragging) {{
        angle += 0.008;
        figure.rotation.y = Math.sin(angle) * 0.45;
    }}
    renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""
