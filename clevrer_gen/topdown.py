"""Top-down (bird's-eye) renderer -- a standalone parallel pipeline.

Independent of render.py/generate.py: it re-simulates a scene deterministically
from the same config (seed + index) and renders it through an *orthographic*
camera looking straight down, like Miniworld's top view. Reuses only the stable
shared helpers (physics, materials, properties, config), so changes here don't
affect the main RGB pipeline and vice versa.

    python render_topdown.py --scenario configs/scenarios/two_collide.yaml \
        --num_videos 5 --output_root output
-> writes <root>/topdown/<split>/video_<a>-<b>/video_<i>.mp4
"""
import math
import os
import shutil
import tempfile

import bpy
import imageio.v2 as imageio
import numpy as np

from . import materials
from .config import ensure_root, resolve_seed
from .physics import build_objects, simulate


def _group_dir(index):
    lo = (index // 1000) * 1000
    return f'video_{lo:05d}-{lo + 1000:05d}'


def topdown_path(root, split, index):
    d = os.path.join(root, 'topdown', split, _group_dir(index))
    return os.path.join(d, f'video_{index:05d}.mp4')


class TopDownRenderer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tcfg = cfg.get('topdown', {}) or {}
        self.res = int(self.tcfg.get('resolution', 320))
        self._build_scene()

    def _build_scene(self):
        cfg, rcfg = self.cfg, self.cfg['render']
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene

        scene.render.engine = 'CYCLES'
        scene.cycles.samples = rcfg['samples']
        scene.cycles.use_denoising = rcfg['denoise']
        scene.render.resolution_x = self.res
        scene.render.resolution_y = self.res
        self._configure_device(rcfg['device'])

        scene.world = bpy.data.worlds.new('World')
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes['Background']
        bg.inputs['Color'].default_value = (0.85, 0.85, 0.85, 1.0)
        bg.inputs['Strength'].default_value = 0.5

        bpy.ops.mesh.primitive_plane_add(size=40)
        ground = bpy.context.object
        ground.name = 'Ground'
        ground.data.materials.append(materials.make_ground_material())

        # Sun slightly off-vertical so objects cast short shadows (depth cue).
        sun = bpy.data.lights.new('Sun', type='SUN')
        sun.energy = 4.0
        sun_ob = bpy.data.objects.new('Sun', sun)
        sun_ob.rotation_euler = (math.radians(12), math.radians(8), 0.0)
        scene.collection.objects.link(sun_ob)

        # Orthographic camera directly above the origin, looking down (-Z).
        cam_data = bpy.data.cameras.new('TopCam')
        cam_data.type = 'ORTHO'
        (xlo, xhi), (ylo, yhi) = cfg['objects']['place_region']
        max_size = max(cfg['properties']['sizes'].values())
        span = max(xhi - xlo, yhi - ylo) + 2 * max_size + \
            float(self.tcfg.get('margin', 1.0))
        cam_data.ortho_scale = float(self.tcfg.get('ortho_scale') or span)
        cam_data.clip_start, cam_data.clip_end = 0.1, 100.0
        cam = bpy.data.objects.new('TopCam', cam_data)
        cam.location = (0.0, 0.0, 20.0)
        cam.rotation_euler = (0.0, 0.0, 0.0)
        scene.collection.objects.link(cam)
        scene.camera = cam

    def _configure_device(self, device):
        scene = bpy.context.scene
        if device == 'CPU':
            scene.cycles.device = 'CPU'
            return
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = device
        prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == device)
        scene.cycles.device = 'GPU'

    def add_objects(self, object_property):
        objs = []
        for spec in object_property:
            s = spec['size_scale']
            shape = spec['shape']
            if shape == 'cube':
                bpy.ops.mesh.primitive_cube_add(size=2 * s)
            elif shape == 'sphere':
                bpy.ops.mesh.primitive_uv_sphere_add(radius=s, segments=48,
                                                     ring_count=24)
            else:
                bpy.ops.mesh.primitive_cylinder_add(radius=s, depth=2 * s,
                                                    vertices=64)
            ob = bpy.context.object
            ob.name = f"obj_{spec['id']:02d}"
            if shape != 'cube':
                bpy.ops.object.shade_smooth()
            ob.data.materials.append(
                materials.make_material(ob.name, spec['color_rgb'], spec['material']))
            ob.rotation_mode = 'QUATERNION'
            objs.append(ob)
        self.objects = objs
        return objs

    def set_pose(self, frame_objs):
        for ob, st in zip(self.objects, frame_objs):
            ob.location = st['location']
            qx, qy, qz, qw = st['orientation']
            ob.rotation_quaternion = (qw, qx, qy, qz)

    def render_frame(self, t, path):
        scene = bpy.context.scene
        scene.frame_set(t)
        scene.render.filepath = path
        scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)


def _mux(frame_paths, out_path, fps):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec='libx264',
                            quality=8, macro_block_size=16) as w:
        for fp in frame_paths:
            w.append_data(imageio.imread(fp))


def render_topdown_video(cfg, index):
    resolve_seed(cfg)
    ensure_root(cfg)
    root, split = cfg['output']['root'], cfg['output']['split']
    out_mp4 = topdown_path(root, split, index)
    if os.path.exists(out_mp4) and not cfg['output']['overwrite']:
        return out_mp4

    rng = np.random.default_rng(cfg['seed'] + index)
    objects = build_objects(cfg, rng)
    sim = simulate(cfg, objects)

    renderer = TopDownRenderer(cfg)
    renderer.add_objects(sim['object_property'])
    tmp = tempfile.mkdtemp(prefix='clevrergen_top_')
    try:
        frame_paths = []
        for t, frame in enumerate(sim['trajectory']):
            renderer.set_pose(frame['objects'])
            fp = os.path.join(tmp, f'{t:05d}.png')
            renderer.render_frame(t, fp)
            frame_paths.append(fp)
        _mux(frame_paths, out_mp4, cfg['video']['fps'])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_mp4


def render_topdown_dataset(cfg, shard_id=0, num_shards=1):
    resolve_seed(cfg)
    ensure_root(cfg)
    start = cfg['output']['start_index']
    indices = [i for i in range(start, start + cfg['output']['num_videos'])
               if (i - start) % num_shards == shard_id]
    print(f'[clevrer-gen:topdown] shard {shard_id}/{num_shards}: '
          f'{len(indices)} videos', flush=True)
    paths = []
    for n, i in enumerate(indices):
        paths.append(render_topdown_video(cfg, i))
        print(f'[clevrer-gen:topdown] [{n + 1}/{len(indices)}] '
              f'video {i} -> {paths[-1]}', flush=True)
    return paths
