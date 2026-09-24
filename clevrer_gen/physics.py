"""Physics stage: place objects from config, simulate with pybullet, and record
a per-frame trajectory plus pairwise collision events.

Runs headless (DIRECT) and has no dependency on bpy, so it is independently
testable. The renderer consumes the returned dict to drive object poses.
"""
import math

import numpy as np
import pybullet as p

from . import properties as props


def _as_range(v):
    """Accept a scalar or [lo, hi]; return (lo, hi)."""
    if isinstance(v, (list, tuple)):
        return float(v[0]), float(v[1])
    return float(v), float(v)


def _sample_speed(v, rng):
    """Sample from a scalar, [lo, hi], or a union of segments [[lo,hi], ...].

    A union draws uniformly over the whole set (segment picked ~ its width).
    """
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
        segs = np.asarray(v, dtype=float)
        widths = segs[:, 1] - segs[:, 0]
        lo, hi = segs[rng.choice(len(segs), p=widths / widths.sum())]
        return rng.uniform(lo, hi)
    lo, hi = _as_range(v)
    return rng.uniform(lo, hi)


def _unit(vec):
    n = math.hypot(vec[0], vec[1])
    return (vec[0] / n, vec[1] / n) if n > 1e-9 else (1.0, 0.0)


def _direction(spec, pos, rng):
    """Resolve an initial in-plane unit direction for one object.
    spec: [vx,vy] explicit | 'toward_center' | 'away_from_center' |
          'x'/'y' (parallel to that axis, random sign) | 'random'."""
    if isinstance(spec, (list, tuple)):
        return _unit(spec)
    if spec == 'toward_center':
        return _unit((-pos[0], -pos[1]))
    if spec == 'away_from_center':
        return _unit((pos[0], pos[1]))
    if spec == 'x':
        return (float(rng.choice([-1.0, 1.0])), 0.0)
    if spec == 'y':
        return (0.0, float(rng.choice([-1.0, 1.0])))
    theta = rng.uniform(0, 2 * math.pi)        # 'random'
    return (math.cos(theta), math.sin(theta))


def _accel(spec, pos, vel, rng):
    """[ax, ay, az] fixed vector, or {magnitude, direction} sampled in-plane per
    object. magnitude: same forms as speed; signed (<0 flips the direction).
    direction: any _direction mode, 'along' (the object's velocity), or a
    {mode: prob} dict to mix modes across objects."""
    if not isinstance(spec, dict):
        return [float(v) for v in spec]
    mode = spec.get('direction', 'random')
    if isinstance(mode, dict):
        modes = list(mode)
        probs = np.asarray([mode[m] for m in modes], dtype=float)
        mode = modes[rng.choice(len(modes), p=probs / probs.sum())]
    mag = _sample_speed(spec['magnitude'], rng)
    dx, dy = _unit(vel) if mode == 'along' else _direction(mode, pos, rng)
    return [float(mag * dx), float(mag * dy), 0.0]


def build_objects(cfg, rng):
    """Resolve the list of object specs (attributes + initial state).

    Honours `objects.explicit` (index-aligned overrides); otherwise samples
    `objects.num` objects with non-overlapping placement.
    """
    ocfg = cfg['objects']
    pcfg = cfg['physics']
    sampler = props.Sampler(cfg['properties'], rng)

    explicit = ocfg.get('explicit')
    if explicit:
        n = len(explicit)
    else:
        num = ocfg['num']
        n = int(rng.integers(num[0], num[1] + 1)) if isinstance(num, (list, tuple)) \
            else int(num)
        explicit = [{} for _ in range(n)]

    (xlo, xhi), (ylo, yhi) = ocfg['place_region']
    spin_lo, spin_hi = _as_range(ocfg['angular_speed'])

    placed = []   # (x, y, radius) for spacing checks
    objects = []
    tries = 0
    while len(objects) < n:
        spec = explicit[len(objects)]
        shape = spec.get('shape') or sampler.shape()
        material = spec.get('material') or sampler.material()
        if 'color' in spec:
            color_name = spec['color']
            color_rgb = cfg['properties']['colors'][color_name]
        else:
            color_name, color_rgb = sampler.color()
        if 'size' in spec:
            size_name = spec['size']
            size_scale = cfg['properties']['sizes'][size_name]
        else:
            size_name, size_scale = sampler.size()

        # Placement: fixed position, a per-object position_range, or the global
        # region. Fixed positions bypass the non-overlap check.
        fixed_pos = 'position' in spec
        if fixed_pos:
            x, y = spec['position']
        elif 'position_range' in spec:
            (rxlo, rxhi), (rylo, ryhi) = spec['position_range']
            x, y = rng.uniform(rxlo, rxhi), rng.uniform(rylo, ryhi)
        else:
            x, y = rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)
        ok = all(math.hypot(x - px, y - py) - size_scale - pr >= ocfg['min_dist']
                 for px, py, pr in placed)
        if not (ok or fixed_pos):
            tries += 1
            if tries > ocfg['max_place_retries']:
                placed, objects, tries = [], [], 0   # restart placement
            continue
        tries = 0

        # Velocity: fixed, or sampled from per-object (falling back to global)
        # speed range + direction mode.
        if 'velocity' in spec:
            vx, vy = spec['velocity']
        else:
            speed = _sample_speed(spec.get('speed', ocfg['speed']), rng)
            dx, dy = _direction(spec.get('direction', ocfg['direction']), (x, y), rng)
            vx, vy = speed * dx, speed * dy
        accel = _accel(spec.get('accel', ocfg['accel']), (x, y), (vx, vy), rng)

        spin = rng.uniform(spin_lo, spin_hi)
        axis = rng.normal(size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        ang_vel = (spin * axis).tolist()

        placed.append((x, y, size_scale))
        objects.append({
            'id': len(objects),
            'shape': shape,
            'color': color_name,
            'color_rgb': color_rgb,
            'material': material,
            'size': size_name,
            'size_scale': size_scale,
            'mass': float(spec.get('mass', pcfg['default_mass'])),
            'friction': float(spec.get('friction', pcfg['default_friction'])),
            'restitution': float(spec.get('restitution', pcfg['default_restitution'])),
            'position': [float(x), float(y), props.rest_height(shape, size_scale)],
            'velocity': [float(vx), float(vy), 0.0],
            'accel': accel,
            'angular_velocity': ang_vel,
        })
    return objects


def _make_body(obj, pcfg):
    """Create a pybullet rigid body for one object spec; return its body id."""
    kind, dim = props.half_extents(obj['shape'], obj['size_scale'])
    if kind == 'box':
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
    elif kind == 'sphere':
        cid = p.createCollisionShape(p.GEOM_SPHERE, radius=dim)
    else:
        radius, half_h = dim
        cid = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=2 * half_h)
    body = p.createMultiBody(baseMass=obj['mass'], baseCollisionShapeIndex=cid,
                             basePosition=obj['position'])
    p.changeDynamics(body, -1, lateralFriction=obj['friction'],
                     restitution=obj['restitution'],
                     linearDamping=pcfg['linear_damping'],
                     angularDamping=pcfg['angular_damping'])
    p.resetBaseVelocity(body, linearVelocity=obj['velocity'],
                        angularVelocity=obj['angular_velocity'])
    return body


def _record(body):
    pos, orn = p.getBasePositionAndOrientation(body)
    lin, ang = p.getBaseVelocity(body)
    return {'location': list(pos), 'orientation': list(orn),
            'velocity': list(lin), 'angular_velocity': list(ang)}


def simulate(cfg, objects):
    """Run the simulation and return {object_property, trajectory, collisions}."""
    pcfg = cfg['physics']
    vcfg = cfg['video']
    dt = 1.0 / (vcfg['fps'] * pcfg['substeps'])

    p.connect(p.DIRECT)
    try:
        p.setGravity(*pcfg['gravity'])
        p.setTimeStep(dt)
        ground_col = p.createCollisionShape(p.GEOM_PLANE)
        ground = p.createMultiBody(0, ground_col)
        p.changeDynamics(ground, -1, lateralFriction=pcfg['ground_friction'],
                         restitution=pcfg['default_restitution'])

        bodies = [_make_body(o, pcfg) for o in objects]
        body_to_idx = {b: i for i, b in enumerate(bodies)}
        forces = [(np.asarray(o['accel']) * o['mass']).tolist() for o in objects]

        # Optional settling (not recorded), e.g. to drop objects onto the plane.
        for _ in range(pcfg['settle_frames'] * pcfg['substeps']):
            p.stepSimulation()

        trajectory = [{'frame': 0,
                       'objects': [_record(b) for b in bodies]}]
        collisions = []
        in_contact = set()
        for t in range(1, vcfg['num_frames']):
            for _ in range(pcfg['substeps']):
                for f, b in zip(forces, bodies):
                    if any(f):
                        p.applyExternalForce(b, -1, f, [0, 0, 0], p.WORLD_FRAME)
                p.stepSimulation()
                current = set()
                for c in p.getContactPoints():
                    a, b = c[1], c[2]
                    if a in body_to_idx and b in body_to_idx:
                        pair = tuple(sorted((body_to_idx[a], body_to_idx[b])))
                        current.add(pair)
                        if pair not in in_contact:
                            collisions.append({'frame': t,
                                               'object_ids': list(pair),
                                               'location': list(c[5])})
                in_contact = current
            trajectory.append({'frame': t,
                               'objects': [_record(b) for b in bodies]})
    finally:
        p.disconnect()

    return {
        'object_property': [{k: o[k] for k in
                             ('id', 'shape', 'color', 'color_rgb', 'material',
                              'size', 'size_scale', 'mass', 'accel')} for o in objects],
        'trajectory': trajectory,
        'collisions': collisions,
    }
