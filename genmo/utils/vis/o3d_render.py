import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch

from .renderer_tools import checkerboard_geometry


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"
    LINE = "unlitLine"
    Transparency = "defaultLitTransparency"
    LitSSR = "defaultLitSSR"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":[]
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0,
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0,
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0,
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0,
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0,
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0,
        },
        "Transparency": {
            "metallic": 0.0,
            "roughness": 0.0,
            "reflectance": 0.0,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0,
        },
    }

    def __init__(self):
        # self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.prefab = Settings.DEFAULT_MATERIAL_NAME
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = True
        self.show_ground_plane = True
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord(),
            Settings.LINE: rendering.MaterialRecord(),
            Settings.Transparency: rendering.MaterialRecord(),
            Settings.LitSSR: rendering.MaterialRecord(),
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.LINE].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LINE].shader = Settings.LINE
        self._materials[Settings.LINE].line_width = 3
        self._materials[Settings.Transparency].base_color = [0.467, 0.467, 0.467, 0.2]
        self._materials[Settings.Transparency].base_color = [0.9, 0.9, 0.9, 0.5]
        self._materials[Settings.Transparency].shader = Settings.Transparency
        self._materials[Settings.Transparency].thickness = 1.0
        self._materials[Settings.Transparency].transmission = 1.0
        self._materials[Settings.Transparency].absorption_distance = 10
        self._materials[Settings.Transparency].absorption_color = [0.5, 0.5, 0.5]
        self._materials[Settings.LitSSR].base_color = [0.467, 0.467, 0.467, 0.2]
        self._materials[Settings.LitSSR].shader = Settings.LitSSR
        self._materials[Settings.LitSSR].thickness = 1.0
        self._materials[Settings.LitSSR].transmission = 1.0
        self._materials[Settings.LitSSR].absorption_distance = 10
        self._materials[Settings.LitSSR].absorption_color = [0.5, 0.5, 0.5]
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.LIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        # assert (self.material.shader == Settings.LIT)
        self.prefab = name
        for key, val in Settings.PREFAB[name].items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        """
        It takes a string as an argument, and then sets the attributes of the object to the values in the
        dictionary that is associated with that string

        Args:
          name: The name of the lighting profile.
        """
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


def get_ground(length, center_x, center_z):
    length, center_x, center_z = map(float, (length, center_x, center_z))
    v, f, vc, fc = map(
        torch.from_numpy,
        checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y"),
    )
    # v, f, vc = v.to(device), f.to(device), vc.to(device)
    ground_geometry = [v, f, vc]
    return ground_geometry


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts.cpu().numpy()),
        triangles=o3d.utility.Vector3iVector(faces.cpu().numpy()),
    )
    mesh.compute_vertex_normals()
    if len(colors.shape) == 1:
        colors = colors[None, :].repeat(len(verts), 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

    return mesh
