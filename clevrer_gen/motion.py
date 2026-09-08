"""Dense 2D motion labels from poses + object-id masks + camera (never estimated).

Per pixel: cast a ray, intersect the owning object in its local frame, re-place
that material point with the pose at t+1, reproject. Local-frame intersection
makes rotation exact. Background is static, so its displacement is exactly zero.

Visibility, foreground and background alike: content is visible at t+1 iff its
destination is in frame and still has the same owner in the t+1 id map. That
covers occlusion, leaving frame, and background being covered.
"""
import numpy as np


_T_EPS = 1e-6


def quat_to_matrix(q_xyzw):
    """pybullet-order (x, y, z, w) quaternion -> 3x3 rotation matrix."""
    x, y, z, w = q_xyzw
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    return np.array([
        [1.0 - (y * ys + z * zs), x * ys - w * zs, x * zs + w * ys],
        [x * ys + w * zs, 1.0 - (x * xs + z * zs), y * zs - w * xs],
        [x * zs - w * ys, y * zs + w * xs, 1.0 - (x * xs + y * ys)],
    ])


def object_geometry(obj):
    """(kind, local half-extents, bounding radius) for one object_property entry.

    Reads `render_half_extents` because the masks come from the render, and the
    render geometry has not always matched the pybullet body (see
    Renderer.add_objects). Taking it from the annotation rather than recomputing
    it keeps clips rendered under either convention correct.
    """
    ext = np.asarray(obj['render_half_extents'], dtype=np.float64)
    shape = obj['shape']
    if shape == 'sphere':
        return 'sphere', ext, float(ext[0])
    if shape == 'cube':
        return 'box', ext, float(np.linalg.norm(ext))
    if shape == 'cylinder':
        return 'cylinder', ext, float(np.hypot(ext[0], ext[2]))
    raise ValueError(f'unknown shape: {shape}')


# ---- ray/shape intersection, in the object's local frame --------------------

def _hit_sphere(o, d, radius):
    b = np.einsum('ij,ij->i', d, o)
    c = np.einsum('ij,ij->i', o, o) - radius * radius
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return t, (disc >= 0.0) & (t > _T_EPS)


def _hit_box(o, d, half):
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / np.where(np.abs(d) < 1e-15, 1e-15, d)
        t1, t2 = (-half - o) * inv, (half - o) * inv
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    return tmin, (tmax >= np.maximum(tmin, 0.0)) & (tmin > _T_EPS)


def _hit_cylinder(o, d, radius, half_h):
    """Finite cylinder along local +Z, caps included."""
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2.0 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1])
    c = o[:, 0] ** 2 + o[:, 1] ** 2 - radius * radius
    best = np.full(len(o), np.inf)

    disc = b * b - 4.0 * a * c
    side = (disc >= 0.0) & (a > 1e-15)
    if side.any():
        sq = np.sqrt(np.maximum(disc[side], 0.0))
        for t in ((-b[side] - sq) / (2.0 * a[side]),
                  (-b[side] + sq) / (2.0 * a[side])):
            z = o[side, 2] + t * d[side, 2]
            ok = (t > _T_EPS) & (np.abs(z) <= half_h)
            best[side] = np.minimum(best[side], np.where(ok, t, np.inf))

    for zc in (half_h, -half_h):
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (zc - o[:, 2]) / np.where(np.abs(d[:, 2]) < 1e-15, 1e-15, d[:, 2])
        px, py = o[:, 0] + t * d[:, 0], o[:, 1] + t * d[:, 1]
        ok = (t > _T_EPS) & (px * px + py * py <= radius * radius)
        best = np.minimum(best, np.where(ok, t, np.inf))

    return best, np.isfinite(best)


def intersect_local(origins, dirs, kind, extents, bound_radius):
    """Ray/shape hit -> (t, hit_mask).

    Misses fall back to the bounding sphere's grazing point. They only occur on
    silhouette pixels, where the mask and the analytic surface disagree by under
    a pixel, so this labels edges instead of leaving holes.
    """
    if kind == 'sphere':
        t, hit = _hit_sphere(origins, dirs, extents[0])
    elif kind == 'box':
        t, hit = _hit_box(origins, dirs, extents)
    else:
        t, hit = _hit_cylinder(origins, dirs, extents[0], extents[2])

    if not hit.all():
        b = np.einsum('ij,ij->i', dirs, origins)
        c = np.einsum('ij,ij->i', origins, origins) - bound_radius ** 2
        graze = -b - np.sqrt(np.maximum(b * b - c, 0.0))
        t = np.where(hit, t, np.maximum(graze, _T_EPS))
    return t, hit


# ---- dense field -----------------------------------------------------------

def id_maps(proposals, n_frames, height, width):
    """Per-frame object-id raster from the RLE proposals; background 0, else id+1.

    The source masks are pairwise disjoint, so overwriting is lossless.
    """
    from pycocotools import mask as mask_utils

    maps = np.zeros((n_frames, height, width), dtype=np.int16)
    for frame in proposals['frames']:
        t = frame['frame_index']
        if t >= n_frames:
            continue
        for obj in frame['objects']:
            maps[t][mask_utils.decode(obj['mask']) > 0] = obj['object_id'] + 1
    return maps


def dense_motion(cam, poses, geoms, idmap_t, idmap_next):
    """Per-pixel displacement t -> t+1 in pixels; returns (flow [H,W,2], valid)."""
    H, W = idmap_t.shape
    flow = np.zeros((H, W, 2), dtype=np.float64)
    dst = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(np.float64)
    dst += 0.5   # pixel centres

    for j, geom in enumerate(geoms):
        sel = idmap_t == j + 1
        if not sel.any():
            continue
        ys, xs = np.nonzero(sel)
        px, py = xs + 0.5, ys + 0.5
        (c0, R0), (c1, R1) = poses[j]

        dirs = cam.pixel_rays(px, py)
        o_loc = np.broadcast_to((cam.location - c0) @ R0, dirs.shape)
        d_loc = dirs @ R0

        kind, extents, bound = geom
        t_hit, _ = intersect_local(o_loc, d_loc, kind, extents, bound)
        x_local = o_loc + t_hit[:, None] * d_loc

        p_next, depth = cam.project(c1 + x_local @ R1.T)
        flow[ys, xs] = p_next - np.stack([px, py], axis=-1)
        dst[ys, xs] = np.where((depth > 0)[:, None], p_next, np.nan)

    ix, iy = np.rint(dst[..., 0] - 0.5), np.rint(dst[..., 1] - 0.5)
    in_frame = np.isfinite(ix) & np.isfinite(iy) & \
        (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ixc = np.clip(np.nan_to_num(ix), 0, W - 1).astype(np.int32)
    iyc = np.clip(np.nan_to_num(iy), 0, H - 1).astype(np.int32)
    return flow, in_frame & (idmap_next[iyc, ixc] == idmap_t)


def pool_to_latent(flow, valid, idmap, cell):
    """Pool to a latent grid, displacement in cell units -> [3, gh, gw].

    A cell averages only its majority owner's valid pixels, so cells straddling
    an occlusion boundary report the dominant surface rather than a blend.
    """
    H, W = idmap.shape
    gh, gw = H // cell, W // cell
    n = cell * cell

    def blocks(a):
        return a.reshape(gh, cell, gw, cell).transpose(0, 2, 1, 3).reshape(gh, gw, n)

    ids, val = blocks(idmap), blocks(valid)
    fx, fy = blocks(flow[..., 0]), blocks(flow[..., 1])

    counts = np.zeros((gh, gw, int(ids.max()) + 1), dtype=np.int32)
    np.add.at(counts, (np.arange(gh)[:, None, None],
                       np.arange(gw)[None, :, None], ids), 1)
    owner = counts.argmax(axis=-1)

    is_owner = ids == owner[..., None]
    use = is_owner & val
    k = use.sum(-1)
    safe = np.maximum(k, 1)

    out = np.zeros((3, gh, gw), dtype=np.float32)
    out[0] = np.where(k > 0, (fx * use).sum(-1) / safe, 0.0) / cell
    out[1] = np.where(k > 0, (fy * use).sum(-1) / safe, 0.0) / cell
    out[2] = k / np.maximum(is_owner.sum(-1), 1)
    return out


def clip_motion(cam, annotation, proposals, num_frames, cell, height, width):
    """Motion tensor for one clip: [T, 3, H/cell, W/cell], channels (dx, dy, vis).

    Displacement is in latent-cell units; multiply by `cell` for pixels. The last
    frame has no successor pose, so it is zeroed with vis 0 in every subset,
    keeping T uniform and index-aligned with RGB.
    """
    traj = annotation['motion_trajectory']
    geoms = [object_geometry(o) for o in annotation['object_property']]
    maps = id_maps(proposals, num_frames, height, width)

    out = np.zeros((num_frames, 3, height // cell, width // cell), dtype=np.float32)
    for t in range(num_frames - 1):
        poses = []
        for j in range(len(geoms)):
            a, b = traj[t]['objects'][j], traj[t + 1]['objects'][j]
            poses.append(((np.asarray(a['location']), quat_to_matrix(a['orientation'])),
                          (np.asarray(b['location']), quat_to_matrix(b['orientation']))))
        flow, valid = dense_motion(cam, poses, geoms, maps[t], maps[t + 1])
        out[t] = pool_to_latent(flow, valid, maps[t], cell)
    return out
