"""Render stage: procedural CLEVRER scene, RGB + segmentation (IndexOB EXR).
Flow computed analytically from trajectory + masks (no Cycles vector pass):
exact for spheres, approximate for cubes/cylinders (misses rotation-induced motion).
"""
import os

import bpy
import numpy as np
from mathutils import Vector

from . import materials

_CAM_LOC = (7.48, -6.51, 5.34)   # frames the ~[-3,3] ground region like CLEVRER
# Matches clevr-dataset-gen's base_scene.blend light rig exactly: a weak sun
# plus 3-point (key/fill/back) area lighting, low ambient world strength.
_SUN = ('SUN', (11.6608, -6.6280, 25.8232), 0.45, None)
_LIGHTS = [  # (name, location, energy, area size)
    ('Lamp_Key', (6.4467, -2.9052, 4.2584), 78.5398, 0.5),
    ('Lamp_Fill', (-4.6711, -4.0136, 3.0112), 23.5619, 0.5),
    ('Lamp_Back', (-1.1685, 2.6460, 5.8157), 39.2699, 1.0),
]
_WORLD_COLOR = (0.0509, 0.0509, 0.0509, 1.0)
_WORLD_STRENGTH = 0.0


def _clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


class Renderer:
    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.do_seg = cfg['render']['passes']['segmentation']
        self.do_flow = cfg['render']['passes']['flow']
        self.rng = np.random.default_rng(cfg['seed'] if seed is None else seed)
        self._pass_dir = None
        self._build_scene()

    # ---- scene construction -------------------------------------------------
    def _build_scene(self):
        rcfg, vcfg = self.cfg['render'], self.cfg['video']
        _clear_scene()
        scene = bpy.context.scene

        scene.render.engine = 'CYCLES'
        scene.cycles.samples = rcfg['samples']
        scene.cycles.use_denoising = rcfg['denoise']
        scene.render.resolution_x = vcfg['width']
        scene.render.resolution_y = vcfg['height']
        scene.render.resolution_percentage = 100
        self._configure_device(rcfg['device'])
        threads = rcfg.get('threads')
        if threads:
            scene.render.threads_mode = 'FIXED'
            scene.render.threads = threads

        # match CLEVR's Standard (non-filmic) view transform / flat gamma
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0

        scene.world = bpy.data.worlds.new('World')
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes['Background']
        bg.inputs['Color'].default_value = _WORLD_COLOR
        bg.inputs['Strength'].default_value = _WORLD_STRENGTH

        bpy.ops.mesh.primitive_plane_add(size=40)
        ground = bpy.context.object
        ground.name = 'Ground'
        ground.data.materials.append(materials.make_ground_material())

        lj = rcfg['light_jitter']
        sun_kind, sun_loc, sun_energy, _ = _SUN
        sun = bpy.data.lights.new(name='Sun', type=sun_kind)
        sun.energy = sun_energy
        sun_ob = bpy.data.objects.new('Sun', sun)
        sun_ob.location = sun_loc
        scene.collection.objects.link(sun_ob)

        for name, loc, energy, size in _LIGHTS:
            light = bpy.data.lights.new(name=name, type='AREA')
            light.energy = energy
            light.size = size
            ob = bpy.data.objects.new(name, light)
            ob.location = tuple(c + self.rng.uniform(-lj, lj) for c in loc)
            scene.collection.objects.link(ob)

        cam_data = bpy.data.cameras.new('Camera')
        cam_data.lens = 45.0
        cam = bpy.data.objects.new('Camera', cam_data)
        cj = rcfg['camera_jitter']
        cam.location = tuple(c + self.rng.uniform(-cj, cj) for c in _CAM_LOC)
        scene.collection.objects.link(cam)
        scene.camera = cam
        _look_at(cam, (0, 0, 0))
        self.camera = cam

        self._setup_passes()

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

    def _setup_passes(self):
        if not self.do_seg:
            return
        scene = bpy.context.scene
        scene.view_layers[0].use_pass_object_index = True
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        rl = tree.nodes.new('CompositorNodeRLayers')
        comp = tree.nodes.new('CompositorNodeComposite')
        tree.links.new(rl.outputs['Image'], comp.inputs['Image'])
        self._pass_node = tree.nodes.new('CompositorNodeOutputFile')
        self._pass_node.format.file_format = 'OPEN_EXR'
        self._pass_node.format.color_depth = '32'
        slots = self._pass_node.file_slots
        slots.clear()
        slots.new('index_')
        tree.links.new(rl.outputs['IndexOB'], self._pass_node.inputs['index_'])

    # ---- objects ------------------------------------------------------------
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
            ob.pass_index = spec['id'] + 1   # 0 reserved for background/ground
            ob.data.materials.append(
                materials.make_material(ob.name, spec['color_rgb'], spec['material']))
            ob.rotation_mode = 'QUATERNION'
            objs.append(ob)
        self.objects = objs
        return objs

    def set_pose(self, frame_objs):
        for ob, st in zip(self.objects, frame_objs):
            ob.location = st['location']
            qx, qy, qz, qw = st['orientation']   # pybullet xyzw -> blender wxyz
            ob.rotation_quaternion = (qw, qx, qy, qz)

    def animate(self, trajectory):
        """Keyframe the full trajectory (required for correct per-frame rendering)."""
        scene = bpy.context.scene
        scene.frame_start = 0
        scene.frame_end = len(trajectory) - 1
        for t, frame in enumerate(trajectory):
            self.set_pose(frame['objects'])
            for ob in self.objects:
                ob.keyframe_insert(data_path='location', frame=t)
                ob.keyframe_insert(data_path='rotation_quaternion', frame=t)

    def project(self, point):
        """World point -> (px, py, depth, inside_view)."""
        from bpy_extras.object_utils import world_to_camera_view
        scene = bpy.context.scene
        co = world_to_camera_view(scene, self.camera, Vector(point))
        w, h = scene.render.resolution_x, scene.render.resolution_y
        inside = (0.0 <= co.x <= 1.0) and (0.0 <= co.y <= 1.0) and co.z > 0
        return int(round(co.x * w)), int(round((1 - co.y) * h)), float(co.z), inside

    # ---- rendering ----------------------------------------------------------
    def render_frame(self, t, rgb_path, pass_dir):
        """Render frame t: RGB to rgb_path, pass EXRs into pass_dir."""
        scene = bpy.context.scene
        scene.frame_set(t)
        scene.render.filepath = rgb_path
        scene.render.image_settings.file_format = 'PNG'
        if self._pass_dir != pass_dir and (self.do_seg or self.do_flow):
            self._pass_node.base_path = pass_dir
            self._pass_dir = pass_dir
        bpy.ops.render.render(write_still=True)

    def _read_exr(self, path):
        """Read an EXR written by the compositor via bpy (no external codec).
        Returns [H, W, 4] float32 in top-down row order."""
        img = bpy.data.images.load(path)
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        bpy.data.images.remove(img)
        return buf.reshape(h, w, 4)[::-1]   # Blender stores bottom-up

    def read_masks(self, t, n_objects, pass_dir):
        """Return list of n binary [H,W] masks (None if object absent)."""
        idx = np.rint(self._read_exr(
            os.path.join(pass_dir, f'index_{t:04d}.exr'))[..., 0]).astype(np.int32)
        masks = []
        for j in range(n_objects):
            m = (idx == j + 1)
            masks.append(m.astype(np.uint8) if m.any() else None)
        return masks

    def compute_flow(self, t, trajectory, frame_masks):
        """Analytical forward flow [H,W,2] px/frame. Projects each object center
        at t-1 and t to screen space; assigns the delta to its mask pixels."""
        from bpy_extras.object_utils import world_to_camera_view
        scene = bpy.context.scene
        W, H = scene.render.resolution_x, scene.render.resolution_y
        ref = next((m for m in frame_masks[t - 1] if m is not None), None) if t > 0 else None
        flow = np.zeros((*ref.shape, 2) if ref is not None else (H, W, 2), dtype=np.float32)
        if t == 0:
            return flow
        for j, (op, oc) in enumerate(zip(trajectory[t - 1]['objects'], trajectory[t]['objects'])):
            mask = frame_masks[t - 1][j]
            if mask is None:
                continue
            cp = world_to_camera_view(scene, self.camera, Vector(op['location']))
            cc = world_to_camera_view(scene, self.camera, Vector(oc['location']))
            flow[mask > 0, 0] = (cc.x - cp.x) * W
            flow[mask > 0, 1] = -(cc.y - cp.y) * H
        return flow
