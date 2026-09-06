"""
Step 9 (stretch) - a 3D mannequin preview of a matched outfit.

Deliberately procedural, not a downloaded 3D model file: the whole
figure (head, torso, arms, legs) is built out of basic Three.js shapes
in-browser. A downloaded .glTF mannequin was the original plan (see
outfit-pairing-ai-MASTER.md Section 12), but building it from primitives
instead means zero extra asset weight against the already-tight ~725MB/
1GB Streamlit Cloud memory budget (Section 7) - this renders entirely in
the visitor's browser via JavaScript, using no Python/server memory at
all beyond the small HTML string itself.
"""

from color_swatches import to_hex
from pairing_rules import TOPS, BOTTOMS

_THREE_JS_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"


def resolve_body_colors(category_a, color_a, category_b, color_b):
    """
    Works out which colour goes on the torso/arms vs the legs, given the
    uploaded item's (category, colour) and the matched item's (category,
    colour) - in whichever order they happen to come in, since either one
    could be the top or the bottom. A dress colours the whole figure.
    Falls back to plain gray for a role neither item fills (e.g. a
    blazer matched against a dress leaves the legs undetermined), rather
    than crashing.
    """
    torso_color, legs_color = "gray", "gray"
    for category, color in ((category_a, color_a), (category_b, color_b)):
        if category == "dress":
            torso_color, legs_color = color, color
        elif category in TOPS or category == "blazer":
            torso_color = color
        elif category in BOTTOMS:
            legs_color = color
    return torso_color, legs_color


def render_mannequin_html(torso_color_name, legs_color_name, height=430):
    """
    Returns a full HTML document (for st.components.v1.html) showing a
    simple humanoid figure: torso/arms in torso_color_name, legs in
    legs_color_name. Auto-rotates, and can also be dragged.
    """
    torso_hex = to_hex(torso_color_name)
    legs_hex = to_hex(legs_color_name)

    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin: 0; overflow: hidden; background: transparent; }}
  #label {{
    position: absolute; bottom: 8px; width: 100%; text-align: center;
    font-family: sans-serif; font-size: 12px; color: #888;
  }}
</style>
</head>
<body>
<div id="label">Drag to rotate</div>
<script type="module">
import * as THREE from "{_THREE_JS_CDN}";

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, window.innerWidth / {height}, 0.1, 100);
camera.position.set(0, 0.3, 6);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setSize(window.innerWidth, {height});
document.body.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
keyLight.position.set(2, 3, 4);
scene.add(keyLight);

const skinColor = 0xd8a878;
const torsoColor = 0x{torso_hex[1:]};
const legsColor = 0x{legs_hex[1:]};

const figure = new THREE.Group();

// Head
const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.35, 24, 24),
    new THREE.MeshStandardMaterial({{ color: skinColor }})
);
head.position.y = 2.15;
figure.add(head);

// Torso (capsule = a rounded box shape, reads as a body better than a plain cube)
const torso = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.55, 1.1, 8, 16),
    new THREE.MeshStandardMaterial({{ color: torsoColor }})
);
torso.position.y = 1.15;
figure.add(torso);

// Arms
for (const side of [-1, 1]) {{
    const arm = new THREE.Mesh(
        new THREE.CapsuleGeometry(0.14, 1.05, 8, 16),
        new THREE.MeshStandardMaterial({{ color: torsoColor }})
    );
    arm.position.set(side * 0.72, 1.15, 0);
    arm.rotation.z = side * 0.12;
    figure.add(arm);
}}

// Legs
for (const side of [-1, 1]) {{
    const leg = new THREE.Mesh(
        new THREE.CapsuleGeometry(0.22, 1.3, 8, 16),
        new THREE.MeshStandardMaterial({{ color: legsColor }})
    );
    leg.position.set(side * 0.26, -0.55, 0);
    figure.add(leg);
}}

scene.add(figure);

let dragging = false, lastX = 0;
renderer.domElement.addEventListener("pointerdown", (e) => {{ dragging = true; lastX = e.clientX; }});
window.addEventListener("pointerup", () => dragging = false);
window.addEventListener("pointermove", (e) => {{
    if (dragging) {{
        figure.rotation.y += (e.clientX - lastX) * 0.01;
        lastX = e.clientX;
    }}
}});

function animate() {{
    requestAnimationFrame(animate);
    if (!dragging) figure.rotation.y += 0.004;
    renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""
