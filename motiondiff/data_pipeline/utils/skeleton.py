import math
import re

import numpy as np
from lxml.etree import Element, ElementTree, SubElement, XMLParser, parse
from scipy.spatial.transform import Rotation

from .bvh import Bvh


class Bone:
    def __init__(self):
        # original bone info
        self.id = None
        self.name = None
        self.orient = np.identity(3)
        self.dof_index = []
        self.channels = []  # bvh only
        self.lb = []
        self.ub = []
        self.parent = None
        self.child = []

        # asf specific
        self.dir = np.zeros(3)
        self.len = 0
        # bvh specific
        self.offset = np.zeros(3)

        # inferred info
        self.pos = np.zeros(3)
        self.end = np.zeros(3)

    def __repr__(self):
        return f"{self.name}"


class Skeleton:
    def __init__(self):
        self.bones = []
        self.name2bone = {}
        self.mass_scale = 1.0
        self.len_scale = 1.0
        self.dof_name = ["x", "y", "z"]
        self.root = None

    def get_parent_indices(self):
        parent_indices = [-1] * len(self.bones)
        for bone in self.bones:
            if bone.parent:
                parent_indices[bone.id] = bone.parent.id
        return parent_indices

    def get_neutral_joints(self):
        joints = []
        for bone in self.bones:
            joints.append(bone.pos)
        joints = np.stack(joints, axis=0)
        return joints

    def load_from_bvh(self, fname, exclude_bones=None, spec_channels=None):
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()
        with open(fname) as f:
            mocap = Bvh(f.read())

        joint_names = list(
            filter(
                lambda x: all([t not in x for t in exclude_bones]),
                mocap.get_joints_names(),
            )
        )
        dof_ind = {"x": 0, "y": 1, "z": 2}
        self.len_scale = 1.0
        self.root = Bone()
        self.root.id = 0
        self.root.name = joint_names[0]
        self.root.channels = mocap.joint_channels(self.root.name)
        self.root.offset = np.array(mocap.joint_offset(self.root.name)) * self.len_scale
        self.name2bone[self.root.name] = self.root
        self.bones.append(self.root)
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = joint
            bone.channels = (
                spec_channels[joint]
                if joint in spec_channels.keys()
                else mocap.joint_channels(joint)
            )
            bone.dof_index = [dof_ind[x[0].lower()] for x in bone.channels]
            bone.offset = np.array(mocap.joint_offset(joint)) * self.len_scale
            bone.lb = [-180.0] * 3
            bone.ub = [180.0] * 3
            self.bones.append(bone)
            self.name2bone[joint] = bone

        # for bone in self.bones:
        #     print(bone.name, bone.channels, bone.offset)

        for bone in self.bones[1:]:
            parent_name = mocap.joint_parent(bone.name).name
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p

        self.forward_bvh(self.root)
        for bone in self.bones:
            if len(bone.child) == 0:
                bone.end = (
                    bone.pos
                    + np.array(
                        [
                            float(x)
                            for x in mocap.get_joint(bone.name).children[-1]["OFFSET"]
                        ]
                    )
                    * self.len_scale
                )
            else:
                bone.end = sum([bone_c.pos for bone_c in bone.child]) / len(bone.child)

    def forward_bvh(self, bone):
        if bone.parent:
            bone.pos = bone.parent.pos + bone.offset
        else:
            bone.pos = bone.offset
        for bone_c in bone.child:
            self.forward_bvh(bone_c)

    def write_xml(self, fname, template_fname):
        parser = XMLParser(remove_blank_text=True)
        tree = parse(template_fname, parser=parser)
        worldbody = tree.getroot().find("worldbody")
        self.write_xml_bodynode(self.root, worldbody)

        # create actuators
        actuators = tree.getroot().find("actuator")
        joints = worldbody.findall(".//joint")
        for joint in joints[1:]:
            name = joint.attrib["name"]
            attr = dict()
            attr["name"] = name
            attr["joint"] = name
            attr["gear"] = "1"
            SubElement(actuators, "motor", attr)

        tree.write(fname, pretty_print=True)

    def write_xml_bodynode(self, bone, parent_node):
        attr = dict()
        attr["name"] = bone.name
        attr["user"] = "{0:.4f} {1:.4f} {2:.4f}".format(*bone.end)
        node = SubElement(parent_node, "body", attr)

        # write joints
        if bone.parent is None:
            j_attr = dict()
            j_attr["name"] = bone.name
            j_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*bone.pos)
            j_attr["limited"] = "false"
            j_attr["type"] = "free"
            j_attr["armature"] = "0"
            j_attr["damping"] = "0"
            j_attr["stiffness"] = "0"
            SubElement(node, "joint", j_attr)
        else:
            for i in range(len(bone.dof_index)):
                ind = bone.dof_index[i]
                axis = bone.orient[:, ind]
                j_attr = dict()
                j_attr["name"] = bone.name + "_" + self.dof_name[ind]
                j_attr["type"] = "hinge"
                j_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*bone.pos)
                j_attr["axis"] = "{0:.4f} {1:.4f} {2:.4f}".format(*axis)
                if i < len(bone.lb):
                    j_attr["range"] = "{0:.4f} {1:.4f}".format(bone.lb[i], bone.ub[i])
                else:
                    j_attr["range"] = "-180.0 180.0"
                SubElement(node, "joint", j_attr)

        # write geometry
        if bone.parent is None:
            g_attr = dict()
            g_attr["size"] = "0.03"
            g_attr["type"] = "sphere"
            g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*bone.pos)
            SubElement(node, "geom", g_attr)
        else:
            e1 = bone.pos.copy()
            e2 = bone.end.copy()
            v = e2 - e1
            if np.linalg.norm(v) > 1e-6:
                v /= np.linalg.norm(v)
            else:
                v = np.array([0.0, 0.0, 0.2])
            e1 += v * 0.02
            e2 -= v * 0.02
            g_attr = dict()
            g_attr["size"] = "0.03"
            g_attr["type"] = "capsule"
            g_attr["fromto"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}".format(
                *np.concatenate([e1, e2])
            )
            SubElement(node, "geom", g_attr)

        # write child bones
        for bone_c in bone.child:
            self.write_xml_bodynode(bone_c, node)


def load_bvh_animation(fname, skeleton):
    with open(fname) as f:
        mocap = Bvh(f.read())

    root_trans = np.array(
        mocap.frames_joint_channels(
            skeleton.root.name, ["Xposition", "Yposition", "Zposition"]
        )
    )

    joint_eulers = []
    for bone in skeleton.bones:
        # print(bone.id, bone.name)
        euler = np.deg2rad(
            np.array(
                mocap.frames_joint_channels(
                    bone.name, ["Zrotation", "Xrotation", "Yrotation"]
                )
            )
        )
        joint_eulers.append(euler)
    joint_eulers = np.stack(joint_eulers, axis=1)

    rotations = Rotation.from_euler("ZXY", joint_eulers.reshape(-1, 3))
    joint_rot_mats = rotations.as_matrix().reshape(joint_eulers.shape[:-1] + (3, 3))

    return root_trans, joint_rot_mats
