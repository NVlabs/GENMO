import sys
if __name__ == '__main__':
    sys.path.append('./')
import pyvista
import time
import math
import numpy as np
import os
import os.path as osp
import shutil
import platform
import tempfile
from .pyrender_camera import IntrinsicsCamera

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import cv2 as cv
from .vis import images_to_video, make_checker_board_texture, nparray_to_vtk_matrix


def start_xvfb(wait=3, window_size=None, display_num=':99'):
    """Start the virtual framebuffer Xvfb.

    Parameters
    ----------
    wait : float, optional
        Time to wait for the virtual framebuffer to start.  Set to 0
        to disable wait.

    window_size : list, optional
        Window size of the virtual frame buffer.  Defaults to
        :attr:`pyvista.global_theme.window_size
        <pyvista.themes.DefaultTheme.window_size>`.

    Notes
    -----
    Only available on Linux.  Be sure to install ``libgl1-mesa-glx
    xvfb`` in your package manager.

    Examples
    --------
    >>> import pyvista
    >>> pyvista.start_xvfb()  # doctest:+SKIP

    """
    from pyvista import global_theme

    if os.name != 'posix':
        raise OSError('`start_xvfb` is only supported on Linux')

    # use current default window size
    if window_size is None:
        window_size = global_theme.window_size
    window_size_parm = f'{window_size[0]:d}x{window_size[1]:d}x24'
    os.system(f'Xvfb {display_num} -screen 0 {window_size_parm} > /dev/null 2>&1 &')
    os.environ['DISPLAY'] = display_num
    if wait:
        time.sleep(wait)


class Visualizer3D:

    def __init__(self, init_T=6, enable_shadow=False, anti_aliasing=False, use_floor=True,
                 add_cube=False, distance=5, elevation=20, azimuth=0, verbose=True, display_num=':99'):
        self.enable_shadow = enable_shadow
        self.anti_aliasing = anti_aliasing
        self.use_floor = use_floor
        self.add_cube = add_cube
        self.verbose = verbose
        self.pl = None
        self.interactive = True
        self.hide_env = False
        # animation control
        self.fr = 0
        self.num_fr = 1
        self.fps_arr = [10, 20, 30, 40, 50, 60]
        self.T_arr = [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60]
        self.T = init_T
        self.paused = False
        self.reverse = False
        self.repeat = False
        # camera
        self.distance = distance
        self.elevation = elevation
        self.azimuth = azimuth
        # background
        self.background_img = None
        # screencaps
        self.take_screenshot = False
        self.recording_video = False
        self.recording_dir = None
        self.recording_fr = 0
        self.xvfb_started = False
        self.display_num = display_num
    
    def init_camera(self):
        # self.pl.camera_position = 'yz'
        self.pl.camera.focal_point = (0, 0, 0)
        self.pl.camera.position = (self.distance, 0, 0)
        self.pl.camera.elevation = self.elevation
        self.pl.camera.azimuth = self.azimuth
        # self.pl.camera.zoom(1.0)

    def set_camera_instrinsics(self, fx=None, fy=None, cx=None, cy=None, z_near=0.1, zfar=1000):
        wsize =  np.array(self.pl.window_size)
        if platform.system() == 'Darwin':
            wsize //= 2
            if not (fx is None and fy is None):
                fx *= 0.5
                fy *= 0.5

        if fx is None and fy is None:
            fx = fy = wsize.max()
        if cx is None and cy is None:
            cx, cy = 0.5 * wsize

        intrinsic_cam = IntrinsicsCamera(fx, fy, cx, cy, z_near, zfar)
        proj_transform = intrinsic_cam.get_projection_matrix(*self.pl.window_size)
        self.pl.camera.SetExplicitProjectionTransformMatrix(nparray_to_vtk_matrix(proj_transform))
        self.pl.camera.SetUseExplicitProjectionTransformMatrix(1)

    def init_scene(self, init_args):
        if not self.hide_env:
            # self.pl.set_background('#DBDAD9')
            self.pl.set_background('#C9DFFF', top='#C9DFFF')    # Classic Rose -> Lavender Blue
        # shadow
        if self.enable_shadow:
            self.pl.enable_shadows()
        if self.anti_aliasing:
            self.pl.enable_anti_aliasing()
        # floor
        if self.use_floor:
            wlh = (20.0, 20.0, 0.05)
            center = np.array([0, 0, -wlh[2] * 0.5])
            self.floor_mesh = pyvista.Cube(center, *wlh)
            self.floor_mesh.active_t_coords *= 2 / self.floor_mesh.active_t_coords.max()
            tex = pyvista.numpy_to_texture(make_checker_board_texture('#81C6EB', '#D4F1F7', width=1000, height=1000, n_tile=1))
            self.floor_actor = self.pl.add_mesh(self.floor_mesh, texture=tex, ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
        else:
            self.floor_actor = self.floor_mesh = None
        # cube
        if self.add_cube:
            self.cube_mesh = pyvista.Box()
            self.cube_mesh.points *= 0.1
            self.cube_mesh.translate((0.0, 0.0, 0.1))
            self.pl.add_mesh(self.cube_mesh, color='orange', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=10, smooth_shading=True)
        
    def update_camera(self, interactive):
        pass

    def update_scene(self):
        pass

    def setup_key_callback(self):

        def close():
            # exit(0)
            self.animating = False

        def slowdown():
            if self.frame_mode == 'fps':
                self.fps = self.fps_arr[(self.fps_arr.index(self.fps) - 1) % len(self.fps_arr)]
            else:
                self.T = self.T_arr[(self.T_arr.index(self.T) + 1) % len(self.T_arr)]

        def speedup():
            if self.frame_mode == 'fps':
                self.fps = self.fps_arr[(self.fps_arr.index(self.fps) + 1) % len(self.fps_arr)]
            else:
                self.T = self.T_arr[(self.T_arr.index(self.T) - 1) % len(self.T_arr)]

        def reverse():
            self.reverse = not self.reverse

        def repeat():
            self.repeat = not self.repeat

        def pause():
            self.paused = not self.paused

        def next_frame():
            if self.fr < self.num_fr - 1:
                self.fr += 1
            self.update_scene()
        
        def prev_frame():
            if self.fr > 0:
                self.fr -= 1
            self.update_scene()

        def go_to_start():
            self.fr = 0
            self.update_scene()

        def go_to_end():
            self.fr = self.num_fr - 1
            self.update_scene()

        def screenshot():
            self.take_screenshot = True

        def record_video():
            self.recording_video = not self.recording_video
            if self.recording_video:
                print('Starting video recording...')
                self.recording_dir = tempfile.mkdtemp(prefix="visualizer3d-")
                print(f'Saving frames to temporary dir: {self.recording_dir}')
            else:
                vid_save_path = osp.join(osp.expanduser("~"), f'Desktop/visualizer3d_{int(time.time())}.mp4')
                print(f'Saving recorded video to {vid_save_path}...')
                images_to_video(self.recording_dir, vid_save_path, img_fmt="%06d.jpg",
                                fps=self.fps, verbose=self.verbose)
                if osp.exists(self.recording_dir):
                    shutil.rmtree(self.recording_dir)
                self.recording_fr = 0
                self.recording_dir = None

        self.pl.add_key_event('q', close)
        self.pl.add_key_event('s', slowdown)
        self.pl.add_key_event('d', speedup)
        self.pl.add_key_event('a', reverse)
        self.pl.add_key_event('g', repeat)
        self.pl.add_key_event('Up', go_to_start)
        self.pl.add_key_event('Down', go_to_end)
        self.pl.add_key_event('space', pause)
        self.pl.add_key_event('Left', prev_frame)
        self.pl.add_key_event('Right', next_frame)
        self.pl.add_key_event('p', screenshot)
        self.pl.add_key_event('m', record_video)

    def render(self, interactive):
        self.update_camera(interactive)
        self.pl.update()
        
    def tframe_animation_loop(self):
        t = 0
        self.animating = True
        while self.animating:
            self.render(interactive=True)
            if t >= math.floor(self.T):
                if not self.reverse:
                    if self.fr < self.num_fr - 1:
                        self.fr += 1
                    elif self.repeat:
                        self.fr = 0
                elif self.reverse and self.fr > 0:
                    self.fr -= 1
                self.update_scene()
                t = 0
            if not self.paused:
                t += 1

    def fps_animation_loop(self):
        self.update_scene()
        self.animating = True
        last_render_time = 0
        while self.animating:
            # print('fps', 1 / (time.time() - last_render_time))
            self.render(interactive=True)
            last_render_time = time.time()
            self.handle_screencap()

            if not self.paused:
                if not self.reverse:
                    if self.fr < self.num_fr - 1:
                        self.fr += 1
                        self.update_scene()
                    elif self.repeat:
                        self.fr = 0
                        self.update_scene()
                elif self.reverse and self.fr > 0:
                    self.fr -= 1
                    self.update_scene()
            
            while True:
                if time.time() - last_render_time >= (1 / self.fps - 1e-6):
                    break
    
    def show_animation(self, window_size=(800, 800), init_args=None, enable_shadow=None, frame_mode='fps', fps=30, repeat=False, show_axes=True):
        self.interactive = True
        self.frame_mode = frame_mode
        self.fps = fps
        self.repeat = repeat
        if enable_shadow is not None:
            self.enable_shadow = enable_shadow
        self.pl = pyvista.Plotter(window_size=window_size)
        self.init_camera()
        self.init_scene(init_args)
        self.update_scene()
        self.setup_key_callback()
        if show_axes:
            self.pl.show_axes()
        self.pl.show(interactive_update=True)
        if self.frame_mode == 'fps':
            self.fps_animation_loop()
        else:
            self.tframe_animation_loop()

    def save_frame(self, fr, img_path):
        self.fr = fr
        self.update_scene()
        self.render(interactive=False)
        if self.background_img is not None:
            img = self.pl.screenshot(transparent_background=True, return_img=True)
            alpha = img[..., [3]] / 255.0
            alpha[alpha > 0.0] = 1.0
            fg_img = cv.cvtColor(img[..., :3], cv.COLOR_RGB2BGR)
            bg_img = cv.imread(self.background_img)
            c_img = fg_img * alpha + bg_img * (1 - alpha)
            cv.imwrite(img_path, c_img)
        else:
            self.pl.screenshot(img_path)


    def save_animation_as_video(self, video_path, init_args=None, window_size=(800, 800), enable_shadow=None, fps=30, crf=25, frame_dir=None, cleanup=True, no_vid=False, frame_prefix=''):
        if platform.system() == 'Linux' and not self.xvfb_started:
            start_xvfb(display_num=self.display_num)
            self.xvfb_started = True
        self.interactive = False
        if enable_shadow is not None:
            self.enable_shadow = enable_shadow
        self.pl = pyvista.Plotter(window_size=window_size, off_screen=True)
        self.init_camera()
        self.init_scene(init_args)
        self.pl.show(interactive_update=True)
        if frame_dir is None:
            frame_dir = tempfile.mkdtemp(prefix="visualizer3d-")
        else:
            if osp.exists(frame_dir):
                shutil.rmtree(frame_dir)
            os.makedirs(frame_dir)
        os.makedirs(osp.dirname(video_path), exist_ok=True)
        for fr in range(self.num_fr):
            self.save_frame(fr, f'{frame_dir}/{frame_prefix}{fr:06d}.jpg')
        if not no_vid:
            images_to_video(frame_dir, video_path, fps=fps, crf=crf, verbose=self.verbose)
        if cleanup:
            shutil.rmtree(frame_dir)

    def handle_screencap(self):
        if self.take_screenshot:
            self.pl.screenshot(f'~/Desktop/visualizer3d_{int(time.time())}.jpg')
            self.take_screenshot = False
        if self.recording_video:
            img_path = osp.join(self.recording_dir, f'{self.recording_fr:06d}.jpg')
            self.pl.screenshot(img_path)
            self.recording_fr += 1



if __name__ == '__main__':
    visualizer = Visualizer3D(add_cube=True, enable_shadow=True)
    visualizer.show_animation(show_axes=True)