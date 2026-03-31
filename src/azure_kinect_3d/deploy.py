"""Deploy reconstructed mesh to DISCOVERSE MuJoCo scene.

Converts OBJ mesh → MJCF via mesh2mjcf.py, then patches the scene XML
so the object replaces the default cube in the simulation.
"""

import os
import re
import subprocess
from pathlib import Path

DISCOVERSE_ROOT = Path("/home/airbot/Documents/DISCOVERSE")
SCENE_XML = DISCOVERSE_ROOT / "models/mjcf/manipulator/roombia/airplay_pick_blocks.xml"
MESH2MJCF = DISCOVERSE_ROOT / "scripts/mesh2mjcf.py"


def deploy_to_discoverse(obj_path: str) -> bool:
    """Convert OBJ to MJCF and patch the scene XML.

    Args:
        obj_path: Absolute path to the .obj mesh file.

    Returns:
        True on success, False on failure.
    """
    obj_path = os.path.abspath(obj_path)
    asset_name = Path(obj_path).stem  # e.g. "seg_20260328_161544_mesh"

    # ── Step A: mesh2mjcf ─────────────────────────────────────────────────
    print(f"[Deploy] Converting {asset_name} to MJCF ...")
    try:
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        result = subprocess.run(
            [
                "uv", "run", "python", str(MESH2MJCF),
                obj_path,
                "--mass", "0.05",
                "--free_joint",
                "-cd",
            ],
            cwd=str(DISCOVERSE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"[Deploy] mesh2mjcf failed:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"[Deploy] mesh2mjcf error: {e}")
        return False

    # Verify outputs exist
    models_dir = DISCOVERSE_ROOT / "models"
    deps_xml = models_dir / "mjcf/object" / f"{asset_name}_dependencies.xml"
    body_xml = models_dir / "mjcf/object" / f"{asset_name}.xml"
    if not deps_xml.exists() or not body_xml.exists():
        print(f"[Deploy] Expected MJCF files not found: {deps_xml}, {body_xml}")
        return False

    # ── Step B: Patch scene XML ───────────────────────────────────────────
    print("[Deploy] Patching scene XML ...")
    try:
        text = SCENE_XML.read_text(encoding="utf-8")

        # Ensure obj_visual / obj_collision default classes exist
        # (mesh2mjcf generates geoms that reference these classes)
        if 'class="obj_visual"' not in text:
            defaults_block = (
                '  <default>\n'
                '    <default class="obj_visual">\n'
                '      <geom group="2" type="mesh" contype="0" conaffinity="0"/>\n'
                '    </default>\n'
                '    <default class="obj_collision">\n'
                '      <geom group="3" condim="3" solimp="0.99 0.995 0.01"'
                ' solref="0.005 1" friction="1 0.005 0.0001" type="mesh"/>\n'
                '    </default>\n'
                '  </default>\n\n'
            )
            # Insert right after the opening <mujoco ...> tag
            text = re.sub(
                r'(<mujoco[^>]*>)\n',
                r'\1\n' + defaults_block,
                text,
                count=1,
            )

        # Replace dependencies include (any previous object)
        text = re.sub(
            r'<include\s+file="../../object/[^"]*_dependencies\.xml"\s*/>',
            f'<include file="../../object/{asset_name}_dependencies.xml"/>',
            text,
        )

        # Replace body include: either the original cube_freebody include,
        # or a previously deployed body-wrapped include block.
        # Pattern 1: bare include (original cube_freebody)
        text = re.sub(
            r'<include\s+file="../../object/cube_freebody\.xml"\s*/>',
            (
                f'<body name="cube" pos="0.25 0.30 0.35">\n'
                f'      <include file="../../object/{asset_name}.xml"/>\n'
                f'    </body>'
            ),
            text,
        )
        # Pattern 2: previously deployed body-wrapped include
        text = re.sub(
            r'<body name="cube" pos="[^"]*">\s*'
            r'<include file="../../object/[^"]*\.xml"\s*/>\s*'
            r'</body>',
            (
                f'<body name="cube" pos="0.25 0.30 0.35">\n'
                f'      <include file="../../object/{asset_name}.xml"/>\n'
                f'    </body>'
            ),
            text,
        )

        SCENE_XML.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"[Deploy] Failed to patch scene XML: {e}")
        return False

    print(f"[Deploy] Done! Object '{asset_name}' deployed to scene.")
    print(f"[Deploy] Run publish.py to see it in simulation.")
    return True
