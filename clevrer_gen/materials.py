"""Procedural CLEVRER-style materials (no .blend dependency).

rubber -> matte diffuse-ish Principled BSDF; metal -> metallic + low roughness.
"""
import bpy

from . import properties as props


def make_material(name, color_rgb, material_kind):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = props.color_to_rgba(color_rgb)
    if material_kind == 'metal':
        bsdf.inputs['Metallic'].default_value = 1.0
        bsdf.inputs['Roughness'].default_value = 0.2
    else:  # rubber
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.7
    return mat


def make_ground_material():
    mat = bpy.data.materials.new(name='Ground')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs['Roughness'].default_value = 1.0
    return mat
