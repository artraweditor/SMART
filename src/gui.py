import wx
import os
import threading
from pathlib import Path
import numpy as np
from PIL import Image

import config
import denoise

CACHE_BELOW = 0.5   # below this zoom a scaled copy of the image is cached


class ImagePanel(wx.Panel):
    def __init__(self, parent, engine, mode='mask'):
        super().__init__(parent)
        self.mode = mode
        self.image = None       # uint8 HxWx3 (the engine's displayed image)
        self._pil = None        # PIL view sharing the buffer of self.image
        self._cache = None      # scaled copy of the image, for small zooms
        self._cache_key = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.painting = False
        self.input_enabled = True
        self.drag_start = (0, 0)
        self.mouse = None       # last mouse position (screen), heal mode
        self.engine = engine

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda e: None)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        self.Bind(wx.EVT_RIGHT_UP, self.on_right_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_leave)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mousewheel)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(
            wx.Colour(*self.engine.conf.background_color))
        if self.mode == 'heal':
            self.SetCursor(wx.Cursor(wx.CURSOR_BLANK))

    # ------------------------------------------------------------ image
    def set_image(self, arr):
        """Use `arr` (uint8 HxWx3) as the displayed image."""
        self.image = arr
        self._cache = None
        self._cache_key = None
        if arr is not None:
            h, w = arr.shape[:2]
            arr = np.ascontiguousarray(arr)
            self.image = arr
            # NOTE: Pillow copies the buffer, so the engine's in-place
            # updates must be copied over with _sync_pil()
            self._pil = Image.fromarray(arr)
        else:
            self._pil = None

    def _sync_pil(self, box):
        """Copy the changed region of self.image into the PIL view."""
        w, h = self.img_size()
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            return
        part = np.ascontiguousarray(self.image[y0:y1, x0:x1])
        self._pil.paste(Image.fromarray(part), (x0, y0))

    def img_size(self):
        if self.image is None:
            return 0, 0
        h, w = self.image.shape[:2]
        return w, h

    def load_image(self, path):
        self.engine.load_image(path)
        self.set_image(self.engine.displayed_image)
        self.center_image()
        self.zoom_fit()

    def has_image(self):
        return self.image is not None

    def center_image(self):
        if self.has_image():
            panel_w, panel_h = self.GetSize()
            w, h = self.img_size()
            self.offset_x = (panel_w - int(w * self.zoom)) // 2
            self.offset_y = (panel_h - int(h * self.zoom)) // 2

    def _dest_size(self):
        w, h = self.img_size()
        return max(1, int(w * self.zoom)), max(1, int(h * self.zoom))

    def _ensure_cache(self):
        dw, dh = self._dest_size()
        if self._cache is None or self._cache_key != (dw, dh):
            self._cache = np.asarray(
                self._pil.resize((dw, dh), Image.BOX)).copy()
            self._cache_key = (dw, dh)

    def _update_cache(self, box):
        """Recompute the cached scaled copy inside the image box."""
        if self._cache is None:
            return
        w, h = self.img_size()
        dw, dh = self._cache_key
        fx, fy = w / dw, h / dh
        x0, y0, x1, y1 = box
        dx0, dy0 = max(int(x0 / fx) - 1, 0), max(int(y0 / fy) - 1, 0)
        dx1, dy1 = min(int(np.ceil(x1 / fx)) + 1, dw), \
            min(int(np.ceil(y1 / fy)) + 1, dh)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        part = self._pil.resize((dx1 - dx0, dy1 - dy0), Image.BOX,
                                box=(dx0 * fx, dy0 * fy,
                                     min(dx1 * fx, w), min(dy1 * fy, h)))
        self._cache[dy0:dy1, dx0:dx1] = np.asarray(part)

    def _render(self, rx, ry, rw, rh):
        """RGB bytes of the screen rectangle (rx, ry, rw, rh), clipped to the
        image: returns (x, y, w, h, buffer) or None."""
        ox, oy = int(self.offset_x), int(self.offset_y)
        dw, dh = self._dest_size()
        x0, y0 = max(rx, ox), max(ry, oy)
        x1, y1 = min(rx + rw, ox + dw), min(ry + rh, oy + dh)
        if x1 <= x0 or y1 <= y0:
            return None
        if self.zoom < CACHE_BELOW:
            self._ensure_cache()
            part = self._cache[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        else:
            w, h = self.img_size()
            fx, fy = w / dw, h / dh
            box = ((x0 - ox) * fx, (y0 - oy) * fy,
                   (x1 - ox) * fx, (y1 - oy) * fy)
            method = Image.NEAREST if self.zoom >= 2 else Image.BILINEAR
            part = np.asarray(self._pil.resize((x1 - x0, y1 - y0), method,
                                               box=box))
        return x0, y0, x1 - x0, y1 - y0, np.ascontiguousarray(part)

    def refresh_image_box(self, box=None):
        """The image changed inside `box` (x0, y0, x1, y1; None: all)."""
        if self.image is None:
            return
        if box is None:
            w, h = self.img_size()
            box = (0, 0, w, h)
        self._sync_pil(box)
        if self.zoom < CACHE_BELOW and self._cache is not None:
            self._update_cache(box)
        x0, y0 = self.to_screen_coords(box[0], box[1])
        x1, y1 = self.to_screen_coords(box[2], box[3])
        self.RefreshRect(wx.Rect(x0 - 2, y0 - 2, x1 - x0 + 5, y1 - y0 + 5))

    # ------------------------------------------------------------ painting
    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self)
        rect = self.GetUpdateRegion().GetBox()
        if rect.IsEmpty():
            rect = wx.Rect(self.GetClientSize())
        dc.SetClippingRegion(rect)
        dc.Clear()
        if self.has_image():
            r = self._render(rect.x, rect.y, rect.width, rect.height)
            if r is not None:
                x, y, w, h, buf = r
                bmp = wx.Bitmap.FromBuffer(w, h, buf)
                dc.DrawBitmap(bmp, x, y, False)
            if self.mode == 'mask':
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), width=1))
                radius = 4
                colors = [wx.Colour(255, 0, 0), wx.Colour(0, 255, 0)]
                for p, b in zip(self.engine.points, self.engine.labels):
                    sx, sy = self.to_screen_coords(p[0], p[1])
                    dc.SetBrush(wx.Brush(colors[b], wx.BRUSHSTYLE_SOLID))
                    dc.DrawCircle(sx, sy, radius)
            elif self.mouse is not None and self.input_enabled:
                self._draw_brush(dc)

    def _brush_radii(self):
        r = self.engine.radius * self.zoom
        return r, r * max(1.0 - self.engine.feather, 0.02)

    def _draw_brush(self, dc):
        x, y = self.mouse
        outer, inner = self._brush_radii()
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        for rad in (outer, inner):
            dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 3))
            dc.DrawCircle(x, y, int(round(rad)))
            dc.SetPen(wx.Pen(wx.Colour(255, 255, 255), 1))
            dc.DrawCircle(x, y, int(round(rad)))

    def refresh_brush(self, pos=None):
        """Repaint the area of the brush cursor at `pos` (default: current)."""
        pos = pos or self.mouse
        if pos is None:
            return
        r = int(self._brush_radii()[0]) + 4
        self.RefreshRect(wx.Rect(pos[0] - r, pos[1] - r, 2 * r + 1, 2 * r + 1))

    # ------------------------------------------------------------ events
    def _panning(self):
        return not wx.GetKeyState(wx.WXK_SHIFT)

    def _paint_to(self, event):
        x, y = self.to_image_coords_f(event.GetX(), event.GetY())
        box = self.engine.stroke_to(x, y)
        if box is not None:
            self.refresh_image_box(box)

    def _start_painting(self, event, erase):
        self.engine.begin_stroke(erase)
        self.painting = True
        if not self.HasCapture():
            self.CaptureMouse()
        self._paint_to(event)

    def _stop_painting(self):
        if self.painting:
            self.painting = False
            if self.HasCapture():
                self.ReleaseMouse()
            box = self.engine.end_stroke()
            if box is not None:
                self.refresh_image_box(box)

    def on_left_down(self, event):
        if self.image is None or not self.input_enabled:
            return
        if self._panning():
            self.dragging = True
            self.drag_start = (event.GetX(), event.GetY())
            self.CaptureMouse()
        elif self.mode == 'heal':
            self._start_painting(event, False)
        else:
            self.engine.add_point(
                self.to_image_coords(event.GetX(), event.GetY()), True)
            self.set_image(self.engine.displayed_image)
            self.Refresh()

    def on_left_up(self, event):
        if self.painting:
            self._stop_painting()
        elif self.image is not None and self.dragging:
            self.dragging = False
            if self.HasCapture():
                self.ReleaseMouse()
            self.Refresh()

    def on_right_down(self, event):
        if self.image is None or not self.input_enabled or self._panning():
            return
        if self.mode == 'heal':
            self._start_painting(event, True)
        else:
            self.engine.add_point(
                self.to_image_coords(event.GetX(), event.GetY()), False)
            self.set_image(self.engine.displayed_image)
            self.Refresh()

    def on_right_up(self, event):
        if self.painting:
            self._stop_painting()

    def on_motion(self, event):
        if self.mode == 'heal' and self.has_image():
            old = self.mouse
            self.mouse = (event.GetX(), event.GetY())
            if old is not None:
                self.refresh_brush(old)
            self.refresh_brush()
        if self.painting and (event.LeftIsDown() or event.RightIsDown()):
            self._paint_to(event)
        elif self.dragging and event.Dragging() and event.LeftIsDown():
            dx = event.GetX() - self.drag_start[0]
            dy = event.GetY() - self.drag_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.drag_start = (event.GetX(), event.GetY())
            self.Refresh()

    def on_leave(self, event):
        if self.mode == 'heal' and self.mouse is not None:
            old, self.mouse = self.mouse, None
            self.refresh_brush(old)

    def on_mousewheel(self, event):
        rotation = event.GetWheelRotation()
        if self.mode == 'heal':
            # wheel: brush size; shift+wheel: feather (on macOS a shifted
            # wheel can come as a horizontal scroll: either axis is fine)
            if rotation and self.input_enabled:
                self.GetParent().adjust_brush(rotation > 0, event.ShiftDown())
            return
        if rotation > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1
        self.center_image()
        self.Refresh()

    def zoom_in(self, event=None):
        self.zoom *= 1.1
        self.center_image()
        self.Refresh()

    def zoom_out(self, event=None):
        self.zoom /= 1.1
        self.center_image()
        self.Refresh()

    def zoom_1_1(self, event=None):
        self.zoom = 1.0
        self.center_image()
        self.Refresh()

    def zoom_fit(self, event=None):
        if self.has_image():
            panel_w, panel_h = self.GetSize()
            img_w, img_h = self.img_size()
            scale_w = panel_w / img_w
            scale_h = panel_h / img_h
            self.zoom = min(scale_w, scale_h)
            self.offset_x = (panel_w - img_w * self.zoom) // 2
            self.offset_y = (panel_h - img_h * self.zoom) // 2
            self.Refresh()

    def reset(self, clear_image=False):
        self.engine.reset(clear_image)
        self.set_image(self.engine.displayed_image)
        self.Refresh()

    def undo(self):
        self.engine.undo_last()
        self.set_image(self.engine.displayed_image)
        self.Refresh()

    # ------------------------------------------------------------ coords
    def to_image_coords_f(self, x, y):
        w, h = self._dest_size()
        real_w, real_h = self.engine.get_size()
        return ((x - int(self.offset_x)) * real_w / w,
                (y - int(self.offset_y)) * real_h / h)

    def to_image_coords(self, x, y):
        fx, fy = self.to_image_coords_f(x, y)
        return int(round(fx)), int(round(fy))

    def to_screen_coords(self, x, y):
        w, h = self._dest_size()
        real_w, real_h = self.engine.get_size()
        return (int(round(x * w / real_w + int(self.offset_x))),
                int(round(y * h / real_h + int(self.offset_y))))

# end of class ImagePanel


class _Layer:
    def __init__(self, arr):
        self.arr = np.ascontiguousarray(arr)
        self.pil = Image.fromarray(self.arr)
        self.cache = None
        self.key = None


class ComparePanel(ImagePanel):
    """Before/after viewer: the two images share zoom and pan, and a
    draggable divider shows "before" on its left and "after" on its right."""
    GRAB = 6    # pixels around the divider that grab it

    def __init__(self, parent, engine):
        super().__init__(parent, engine, 'denoise')
        self.layers = {}
        self.view = 'split'     # 'split', 'before' or 'after'
        self.split = 0.5        # divider position, fraction of the width
        self.moving_divider = False

    # ------------------------------------------------------------ images
    def set_layers(self, before, after=None):
        self.layers = {}
        self.image = before
        if before is not None:
            self.layers['before'] = _Layer(before)
        if after is not None:
            self.layers['after'] = _Layer(after)
        self.Refresh()

    def load_image(self, path):
        self.engine.load_image(path)
        self.set_layers(self.engine.displayed_image)
        self.split = 0.5
        self.center_image()
        self.zoom_fit()

    def show_result(self):
        """The engine has a new denoised image."""
        self.set_layers(self.engine.before_display,
                        self.engine.after_display)
        self.Refresh()

    def reset(self, clear_image=False):
        self.engine.reset(clear_image)
        self.set_layers(self.engine.displayed_image)

    def undo(self):
        self.reset()

    def set_view(self, view):
        self.view = view
        self.Refresh()

    def _mode(self):
        return self.view if 'after' in self.layers else 'before'

    # ------------------------------------------------------------ painting
    def _split_x(self):
        return int(self.offset_x) + int(self.split * self._dest_size()[0])

    def _render_layer(self, name, rx, ry, rw, rh):
        if rw <= 0 or rh <= 0:
            return None
        layer = self.layers[name]
        self._pil, self._cache, self._cache_key = \
            layer.pil, layer.cache, layer.key
        try:
            return ImagePanel._render(self, rx, ry, rw, rh)
        finally:
            layer.cache, layer.key = self._cache, self._cache_key

    def _render(self, rx, ry, rw, rh):
        mode = self._mode()
        if mode != 'split':
            return self._render_layer(mode, rx, ry, rw, rh)
        sx = self._split_x()
        left = self._render_layer('before', rx, ry, min(rx + rw, sx) - rx, rh)
        x1 = max(rx, sx)
        right = self._render_layer('after', x1, ry, rx + rw - x1, rh)
        if left is None or right is None:
            return left or right
        return (left[0], left[1], left[2] + right[2], left[3],
                np.ascontiguousarray(np.concatenate([left[4], right[4]],
                                                    axis=1)))

    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self)
        rect = self.GetUpdateRegion().GetBox()
        if rect.IsEmpty():
            rect = wx.Rect(self.GetClientSize())
        dc.SetClippingRegion(rect)
        dc.Clear()
        if not self.has_image():
            return
        r = self._render(rect.x, rect.y, rect.width, rect.height)
        if r is not None:
            x, y, w, h, buf = r
            dc.DrawBitmap(wx.Bitmap.FromBuffer(w, h, buf), x, y, False)
        if self._mode() == 'split':
            sx = self._split_x()
            dw, dh = self._dest_size()
            y0, y1 = int(self.offset_y), int(self.offset_y) + dh
            if int(self.offset_x) <= sx <= int(self.offset_x) + dw:
                for color, width in ((wx.Colour(0, 0, 0), 3),
                                     (wx.Colour(255, 255, 255), 1)):
                    dc.SetPen(wx.Pen(color, width))
                    dc.DrawLine(sx, y0, sx, y1)
                cy = (y0 + y1) // 2
                dc.SetBrush(wx.Brush(wx.Colour(255, 255, 255)))
                dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1))
                dc.DrawCircle(sx, cy, 7)

    # ------------------------------------------------------------ events
    def _near_divider(self, x):
        return self._mode() == 'split' and \
            abs(x - self._split_x()) <= self.GRAB

    def _move_divider(self, x):
        dw = self._dest_size()[0]
        self.split = min(max((x - int(self.offset_x)) / dw, 0.0), 1.0)
        self.Refresh()

    def on_left_down(self, event):
        if self.image is None:
            return
        if self._near_divider(event.GetX()):
            self.moving_divider = True
        else:
            self.dragging = True
            self.drag_start = (event.GetX(), event.GetY())
        self.CaptureMouse()

    def on_left_up(self, event):
        if self.moving_divider or self.dragging:
            self.moving_divider = self.dragging = False
            if self.HasCapture():
                self.ReleaseMouse()
            self.Refresh()

    def on_right_down(self, event):
        pass

    def on_right_up(self, event):
        pass

    def on_motion(self, event):
        if self.image is None:
            return
        if self.moving_divider and event.LeftIsDown():
            self._move_divider(event.GetX())
        elif self.dragging and event.Dragging() and event.LeftIsDown():
            self.offset_x += event.GetX() - self.drag_start[0]
            self.offset_y += event.GetY() - self.drag_start[1]
            self.drag_start = (event.GetX(), event.GetY())
            self.Refresh()
        else:
            self.SetCursor(wx.Cursor(
                wx.CURSOR_SIZEWE if self._near_divider(event.GetX())
                else wx.CURSOR_ARROW))

    def on_mousewheel(self, event):
        rotation = event.GetWheelRotation()
        if rotation and self.has_image():
            self.zoom_at(1.1 if rotation > 0 else 1 / 1.1,
                         event.GetX(), event.GetY())

    def zoom_at(self, factor, x, y):
        """Zoom keeping the image point under the screen point (x, y)."""
        ix, iy = self.to_image_coords_f(x, y)
        self.zoom *= factor
        dw, dh = self._dest_size()
        rw, rh = self.engine.get_size()
        self.offset_x = x - ix * dw / rw
        self.offset_y = y - iy * dh / rh
        self.Refresh()

# end of class ComparePanel


class MainFrame(wx.Frame):
    def __init__(self, conf, engine, mode='mask', outdir=None,
                 save_as=None):
        self.mode = mode
        self.save_as = save_as  # fixed default save name (no Open button)
        self.outdir = outdir    # default directory of the save dialog
        title = {'mask': "SMART AI mask builder", 'heal': "SMART AI healing",
                 'denoise': "SMART raw denoising"}[mode]
        super().__init__(None, title=title + " - v" + config.version,
                         size=conf.window_size)

        icon = app_icon()
        if icon is not None:
            self.SetIcon(icon)

        self.engine = engine
        self.filename = None
        self.busy_cursor = None

        self.statusbar = self.CreateStatusBar()
        if mode == 'mask':
            msg = "shift+left click: add positive point; " \
                "shift+right click: add negative point   "
        elif mode == 'denoise':
            key = "Cmd" if wx.Platform == '__WXMAC__' else "Ctrl"
            msg = "drag: pan; drag the divider: compare; wheel: zoom; " \
                f"{key}+Enter: denoise   "
        else:
            key = "Cmd" if wx.Platform == '__WXMAC__' else "Ctrl"
            msg = "shift+left drag: paint; shift+right drag: erase; " \
                f"wheel: brush size; shift+wheel: feather; {key}+Enter: heal   "
        msg_size = self.statusbar.GetTextExtent(msg)
        self.statusbar.SetFieldsCount(2, [-1, msg_size.width])
        self.statusbar.SetStatusText("Ready")
        self.statusbar.SetStatusText(msg, 1)

        vbox = wx.BoxSizer(wx.VERTICAL)

        # Menu Bar
        if wx.Platform == '__WXMAC__':
            menubar = wx.MenuBar()
            self.SetMenuBar(menubar)

        # Toolbar
        toolbar = self.CreateToolBar()
        self.toolbar = toolbar
        d = config.resource_dir() / 'icons'
        sz = wx.Size(32, 32)
        self._toolbar_icons = []

        def icon_colour():
            # Use the effective appearance of the app, not the toolbar
            # background: on macOS the latter does not reflect what the
            # (translucent, natively drawn) toolbar actually looks like.
            try:
                dark = wx.SystemSettings.GetAppearance().IsDark()
            except Exception:
                bg = toolbar.GetBackgroundColour()
                lum = (bg[0]/255.0 * 0.213 + bg[1]/255.0 * 0.715 +
                       bg[2]/255.0 * 0.072)
                dark = lum < 0.5
            if os.environ.get('SMART_DEBUG_ICONS'):
                import sys
                bg = toolbar.GetBackgroundColour()
                print("icon colour: toolbar bg=%s window bg=%s "
                      "SYS_COLOUR_WINDOW=%s IsDark=%s" % (
                          tuple(bg), tuple(self.GetBackgroundColour()),
                          tuple(wx.SystemSettings.GetColour(
                              wx.SYS_COLOUR_WINDOW)), dark),
                      file=sys.stderr, flush=True)
            return b'#ffffff' if dark else b'#000000'

        def svg(pth, repl):
            with open(d / pth, 'rb') as f:
                data = f.read().replace(b'#2a7fff', repl)
                return wx.BitmapBundle.FromSVG(data, sz)

        def add_tool(tool_id, label, name, **kwargs):
            tool = toolbar.AddTool(
                tool_id, label, svg(name, icon_colour()), **kwargs)
            self._toolbar_icons.append((tool, name))
            return tool

        def on_sys_colour_changed(event):
            event.Skip()
            repl = icon_colour()
            for tool, name in self._toolbar_icons:
                tool.SetNormalBitmap(svg(name, repl))
            toolbar.Realize()
            toolbar.Refresh()
        self.Bind(wx.EVT_SYS_COLOUR_CHANGED, on_sys_colour_changed)

        what = "mask" if mode == 'mask' else "image"
        self.busy_controls = []
        tb_open = None
        if save_as is None:
            tb_open = add_tool(wx.ID_OPEN, "Open", 'folder-open.svg',
                shortHelp="Open image...")
        tb_save = add_tool(wx.ID_SAVE, "Save", 'save.svg',
            shortHelp=f"Save {what}..." if save_as is None
            else f"Save {what} to {os.path.basename(save_as)}")
        tb_reset = add_tool(wx.ID_ANY, "Reset", 'undo-all.svg',
            shortHelp={'mask': "Remove all points",
                       'heal': "Revert to the image as loaded",
                       'denoise': "Discard the denoised image"}[mode])
        if mode == 'denoise':
            tb_undo = tb_reset
        else:
            tb_undo = add_tool(wx.ID_ANY, "Undo", 'undo.svg',
                shortHelp="Remove last added point" if mode == 'mask'
                else "Undo last stroke or healing")
        tb_zoom_in = add_tool(wx.ID_ZOOM_IN, "Zoom In", 'magnifier-plus.svg',
            shortHelp="Zoom in")
        tb_zoom_out = add_tool(wx.ID_ZOOM_OUT, "Zoom Out", 'magnifier-minus.svg',
            shortHelp="Zoom out")
        tb_zoom_1_1 = add_tool(wx.ID_ANY, "Zoom 1:1", 'magnifier-1to1.svg',
            shortHelp="Zoom to 100%")
        tb_zoom_fit = add_tool(wx.ID_ANY, "Zoom Fit", 'magnifier-fit.svg',
            shortHelp="Zoom to fit")
        heal_key = "Cmd+Enter" if wx.Platform == '__WXMAC__' else "Ctrl+Enter"
        if mode == 'heal':
            toolbar.AddSeparator()
            self.btn_heal = wx.Button(toolbar, label="Heal")
            self.btn_heal.SetToolTip(
                f"Heal the painted areas ({heal_key})")
            toolbar.AddControl(self.btn_heal)
            self.btn_heal.Bind(wx.EVT_BUTTON, self.on_heal)
            self.busy_controls = [self.btn_heal]
            toolbar.AddSeparator()
            self.spin_radius = self._add_spin(
                toolbar, "Radius", engine.radius, 1, 2000, 1, 0)
            self.spin_feather = self._add_spin(
                toolbar, "Feather", engine.feather, 0, 1, 0.05, 2)
            self.spin_opacity = self._add_spin(
                toolbar, "Opacity", engine.opacity, 0.01, 1, 0.05, 2)
            self.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_brush_changed)
            self.Bind(wx.EVT_TEXT_ENTER, self.on_brush_changed)
            self.Bind(wx.EVT_CHAR_HOOK, self.on_char_hook)
        self.id_denoise = wx.NewIdRef()
        self.id_heal = wx.NewIdRef()
        if mode == 'denoise':
            toolbar.AddSeparator()
            toolbar.AddControl(wx.StaticText(toolbar, label=" Model "))
            self.choice_model = wx.Choice(
                toolbar, choices=list(config.DENOISE_MODELS))
            self.choice_model.SetStringSelection(engine.options.model)
            toolbar.AddControl(self.choice_model)
            self.btn_denoise = wx.Button(toolbar, label="Denoise")
            self.btn_denoise.SetToolTip(f"Denoise the image ({heal_key})")
            toolbar.AddControl(self.btn_denoise)
            toolbar.AddSeparator()
            toolbar.AddControl(wx.StaticText(toolbar, label=" View "))
            self.choice_view = wx.Choice(
                toolbar, choices=["Before / after", "Before", "After"])
            self.choice_view.SetSelection(0)
            toolbar.AddControl(self.choice_view)
            self.busy_controls = [self.choice_model, self.btn_denoise]
            self.btn_denoise.Bind(wx.EVT_BUTTON, self.on_denoise)
            self.choice_view.Bind(wx.EVT_CHOICE, self.on_view_changed)
        toolbar.Realize()

        accel_entries = [
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('s'), tb_save.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('z'), tb_undo.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('r'), tb_reset.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('+'), tb_zoom_in.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('='), tb_zoom_in.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('-'), tb_zoom_out.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('1'), tb_zoom_1_1.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('f'), tb_zoom_fit.GetId()),
        ]
        if tb_open is not None:
            accel_entries.append(
                wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('o'), tb_open.GetId()))
        if mode == 'heal':
            accel_entries += [
                wx.AcceleratorEntry(wx.ACCEL_CTRL, wx.WXK_RETURN,
                                    self.id_heal),
                wx.AcceleratorEntry(wx.ACCEL_CTRL, wx.WXK_NUMPAD_ENTER,
                                    self.id_heal),
            ]
        if mode == 'denoise':
            accel_entries += [
                wx.AcceleratorEntry(wx.ACCEL_CTRL, wx.WXK_RETURN,
                                    self.id_denoise),
                wx.AcceleratorEntry(wx.ACCEL_CTRL, wx.WXK_NUMPAD_ENTER,
                                    self.id_denoise),
            ]
            self.Bind(wx.EVT_MENU, self.on_denoise, id=self.id_denoise)
        accel_table = wx.AcceleratorTable(accel_entries)
        self.SetAcceleratorTable(accel_table)

        if tb_open is not None:
            self.Bind(wx.EVT_MENU, self.on_open_image, tb_open)
            self.Bind(wx.EVT_TOOL, self.on_open_image, tb_open)
        self.Bind(wx.EVT_MENU, self.on_save, tb_save)
        self.Bind(wx.EVT_MENU, self.on_reset, tb_reset)
        if mode != 'denoise':
            self.Bind(wx.EVT_MENU, self.on_undo, tb_undo)
        self.Bind(wx.EVT_MENU, self.panel_zoom_in, tb_zoom_in)
        self.Bind(wx.EVT_MENU, self.panel_zoom_out, tb_zoom_out)
        self.Bind(wx.EVT_MENU, self.panel_zoom_1_1, tb_zoom_1_1)
        self.Bind(wx.EVT_MENU, self.panel_zoom_fit, tb_zoom_fit)

        self.Bind(wx.EVT_TOOL, self.panel_zoom_in, tb_zoom_in)
        self.Bind(wx.EVT_TOOL, self.panel_zoom_out, tb_zoom_out)
        self.Bind(wx.EVT_TOOL, self.panel_zoom_1_1, tb_zoom_1_1)
        self.Bind(wx.EVT_TOOL, self.panel_zoom_fit, tb_zoom_fit)
        self.Bind(wx.EVT_TOOL, self.on_save, tb_save)
        self.Bind(wx.EVT_TOOL, self.on_reset, tb_reset)
        if mode != 'denoise':
            self.Bind(wx.EVT_TOOL, self.on_undo, tb_undo)
        if mode == 'heal':
            self.Bind(wx.EVT_MENU, self.on_heal, id=self.id_heal)

        self.Bind(wx.EVT_MENU, self.on_close, id=wx.ID_EXIT)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.image_panel = ComparePanel(self, engine) if mode == 'denoise' \
            else ImagePanel(self, engine, mode)
        hbox.Add(self.image_panel, proportion=1, flag=wx.EXPAND)

        vbox.Add(hbox, proportion=1, flag=wx.EXPAND)
        self.SetSizer(vbox)
        self.Raise()

    # ------------------------------------------------------------ brush UI
    def _add_spin(self, toolbar, label, value, lo, hi, inc, digits):
        toolbar.AddControl(wx.StaticText(toolbar, label=f" {label} "))
        sc = wx.SpinCtrlDouble(toolbar, value=str(value), min=lo, max=hi,
                               inc=inc, size=wx.Size(80, -1),
                               style=wx.TE_PROCESS_ENTER)
        sc.SetDigits(digits)
        sc.SetValue(value)
        toolbar.AddControl(sc)
        return sc

    def on_brush_changed(self, event):
        e = self.engine
        e.radius = self.spin_radius.GetValue()
        e.feather = self.spin_feather.GetValue()
        e.opacity = self.spin_opacity.GetValue()
        self.image_panel.refresh_brush()
        self.image_panel.SetFocus()
        event.Skip()

    def adjust_brush(self, up, feather):
        """Make the brush bigger/smaller (or, with `feather`, more/less
        feathered), keeping the toolbar controls in sync."""
        panel = self.image_panel
        panel.refresh_brush()
        if feather:
            f = self.engine.feather + (0.05 if up else -0.05)
            self.engine.feather = min(max(f, 0.0), 1.0)
            self.spin_feather.SetValue(self.engine.feather)
        else:
            r = self.engine.radius * (1.2 if up else 1 / 1.2)
            self.engine.radius = min(max(r, 1.0), 2000.0)
            self.spin_radius.SetValue(self.engine.radius)
        panel.refresh_brush()

    def on_char_hook(self, event):
        k = event.GetKeyCode()
        if k in (ord('['), ord(']')):
            self.adjust_brush(k == ord(']'), event.ShiftDown())
        else:
            event.Skip()

    # ------------------------------------------------------------ files
    def check_save(self):
        if self.engine.saved:
            return True
        what = "mask" if self.mode == 'mask' else "image"
        with wx.MessageDialog(self,
                              "You have unsaved changes. "
                              f"Do you want to save the {what} "
                              "before continuing?",
                              "Unsaved Changes",
                              wx.YES_NO | wx.CANCEL | wx.ICON_QUESTION) as dlg:
            result = dlg.ShowModal()
        if result == wx.ID_YES:
            return self.save_mask()
        elif result == wx.ID_NO:
            return True
        else:
            return False

    def on_open_image(self, event):
        if self.engine.busy_now():
            return
        if not self.check_save():
            return
        if self.mode == 'denoise':
            wildcard = "DNG files (*.dng)|*.dng;*.DNG"
        else:
            wildcard = "Image files (*.png;*.jpg;*.jpeg;*.tif;*.tiff)" \
                "|*.png;*.jpg;*.jpeg;*.tif;*.tiff"
        with wx.FileDialog(
                self, "Open Image file",
                wildcard=wildcard,
                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                defaultDir=self.engine.conf.last_dir) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            self.load_image(fd.GetPath())

    def load_image(self, path):
        if not self.check_save():
            return
        try:
            self.image_panel.load_image(path)
            self.filename = Path(self.engine.image_filename)
            self.statusbar.SetStatusText(f"loaded image: {self.filename}")
            self.engine.conf.last_dir = os.path.dirname(path)
        except Exception as e:
            self.image_panel.reset(True)
            wx.MessageDialog(self, f"Error loading image:\n{e}",
                             "Load Error",
                             wx.OK | wx.ICON_ERROR).ShowModal()

    def panel_zoom_in(self, event):
        self.image_panel.zoom_in()

    def panel_zoom_out(self, event):
        self.image_panel.zoom_out()

    def panel_zoom_1_1(self, event):
        self.image_panel.zoom_1_1()

    def panel_zoom_fit(self, event):
        self.image_panel.zoom_fit()

    def save_mask(self):
        if self.filename is None:
            self.statusbar.SetStatusText(
                "No mask to save" if self.mode == 'mask'
                else "No image to save")
            return True
        if self.mode == 'denoise' and self.engine.result is None:
            self.statusbar.SetStatusText(
                "Nothing to save: denoise the image first")
            return True
        if self.save_as is not None and not os.path.exists(self.save_as):
            return self._write_output(self.save_as)
        dd = self.outdir or str(self.filename.parent)
        if self.mode == 'denoise':
            title = "Save Denoised Image"
            tiff = "Linear float TIFF files (*.tif;*.tiff)|*.tif;*.tiff"
            dngw = "Linear DNG files (*.dng)|*.dng"
            if self.engine.default_format == 'dng':
                df = str(self.filename.stem + "_denoised.dng")
                wildcard = dngw + "|" + tiff
            else:
                df = str(self.filename.stem + "_denoised.tif")
                wildcard = tiff + "|" + dngw
        elif self.mode == 'mask':
            title = "Save Mask"
            df = str(self.filename.stem + "_mask.png")
            wildcard = "PNG files (*.png)|*.png|All files (*.*)|*.*"
        else:
            title = "Save Image"
            ext = self.filename.suffix or '.tif'
            df = str(self.filename.stem + "_healed" + ext)
            e = ext.lower().lstrip('.')
            kinds = {'tif': 'TIFF files (*.tif;*.tiff)|*.tif;*.tiff',
                     'tiff': 'TIFF files (*.tif;*.tiff)|*.tif;*.tiff',
                     'png': 'PNG files (*.png)|*.png',
                     'jpg': 'JPEG files (*.jpg;*.jpeg)|*.jpg;*.jpeg',
                     'jpeg': 'JPEG files (*.jpg;*.jpeg)|*.jpg;*.jpeg'}
            wildcard = kinds.get(e, kinds['tif']) + "|All files (*.*)|*.*"
        # NOTE: wxWidgets 3.2.9 on macOS asserts (index out of bounds) in
        # FileDialog.ShowModal() with FD_SAVE and a single wildcard filter,
        # so we add a second one as a workaround
        if self.save_as is not None:    # exists already: ask
            dd, df = os.path.split(self.save_as)
        with wx.FileDialog(self, title,
                           wildcard=wildcard,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                           defaultDir=dd, defaultFile=df) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return True
            path = fd.GetPath()
            if self.mode == 'denoise' and not path.lower().endswith(
                    ('.tif', '.tiff', '.dng')):
                dng_first = self.engine.default_format == 'dng'
                path += '.dng' if (fd.GetFilterIndex() == 1) != dng_first \
                    else '.tif'
            return self._write_output(path)

    def _write_output(self, path):
        try:
            if self.mode == 'mask':
                self.engine.save_mask(path)
                self.statusbar.SetStatusText(f"mask saved to {path}")
            else:
                self.engine.save_image(path)
                self.statusbar.SetStatusText(f"image saved to {path}")
            return True
        except Exception as e:
            wx.MessageDialog(self, f"Error saving "
                             f"{'mask' if self.mode == 'mask' else 'image'}"
                             f":\n{e}",
                             "Save Error",
                             wx.OK | wx.ICON_ERROR).ShowModal()
            return False

    def on_save(self, event):
        if self.engine.busy_now():
            return
        self.save_mask()

    def on_reset(self, event):
        if self.engine.busy_now():
            return
        if self.mode == 'heal' and self.engine.history:
            with wx.MessageDialog(self, "Revert the image to how it was when "
                                  "loaded, discarding all changes?",
                                  "Revert", wx.OK | wx.CANCEL |
                                  wx.ICON_QUESTION) as dlg:
                if dlg.ShowModal() != wx.ID_OK:
                    return
        self.image_panel.reset()

    def on_undo(self, event):
        if self.engine.busy_now():
            return
        self.image_panel.undo()

    # ------------------------------------------------------------ healing
    def _set_busy(self, busy):
        self.image_panel.input_enabled = not busy
        for c in self.busy_controls:
            c.Enable(not busy)
        for tool in range(self.toolbar.GetToolsCount()):
            t = self.toolbar.GetToolByPos(tool)
            if t.IsButton():
                self.toolbar.EnableTool(t.GetId(), not busy)
        if busy:
            self.busy_cursor = wx.BusyCursor()
        else:
            self.busy_cursor = None
            if self.mode == 'heal':
                self.image_panel.SetCursor(wx.Cursor(wx.CURSOR_BLANK))

    def on_heal(self, event):
        import heal_engine
        e = self.engine
        if e.img is None or e.busy_now():
            return
        if not e.has_cover():
            self.statusbar.SetStatusText(
                "Nothing to heal: paint over the image with shift+drag")
            return
        self._set_busy(True)
        self.statusbar.SetStatusText("Healing...")
        state = {'k': 0, 'n': 0, 'out': None}

        def progress(k, n):
            state['k'], state['n'] = k, n

        def finish(kind, *args):
            state['out'] = (kind, args)

        if not e.heal_async(
                lambda p, c: finish('done', p, c),
                lambda ex: finish('error', ex),
                progress):
            self._set_busy(False)
            return
        dlg = wx.ProgressDialog(
            "Healing", "Loading the model...", maximum=1000, parent=self,
            style=wx.PD_APP_MODAL | wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME |
            wx.PD_SMOOTH)
        cancelling = False
        while state['out'] is None:
            k, n = state['k'], state['n']
            if cancelling:
                keep = dlg.Pulse("Cancelling...")
            elif n == 0:
                keep = dlg.Pulse("Loading the model...")
            else:
                keep = dlg.Update(min(int(1000 * k / n), 999),
                                  f"Healing... area {min(k + 1, n)} of {n}")
            if isinstance(keep, tuple):
                keep = keep[0]
            if not keep and not cancelling:
                cancelling = True
                e.cancel()
            wx.MilliSleep(50)
            wx.YieldIfNeeded()
        dlg.Destroy()
        kind, args = state['out']
        if kind == 'done':
            self._heal_done(*args)
        elif isinstance(args[0], heal_engine.Cancelled):
            self._set_busy(False)
            self.statusbar.SetStatusText("Healing cancelled")
        else:
            self._heal_failed(args[0])

    def _heal_done(self, patches, cover):
        box = self.engine.apply_patches(patches, cover)
        self._set_busy(False)
        self.image_panel.refresh_image_box(box)
        self.statusbar.SetStatusText("Healing done")

    def _heal_failed(self, ex):
        import traceback
        traceback.print_exception(ex)
        self._set_busy(False)
        self.statusbar.SetStatusText("Healing failed")
        wx.MessageDialog(self, f"Healing failed:\n{ex}", "Healing error",
                         wx.OK | wx.ICON_ERROR).ShowModal()

    # ------------------------------------------------------------ denoising
    def on_view_changed(self, event):
        self.image_panel.set_view(
            ('split', 'before', 'after')[self.choice_view.GetSelection()])

    def on_denoise(self, event):
        e = self.engine
        if e.raw is None:
            self.statusbar.SetStatusText("Open a DNG file first")
            return
        if e.busy_now():
            return
        model = self.choice_model.GetStringSelection()
        if not self._ensure_model(model):
            return
        self._set_busy(True)
        self.statusbar.SetStatusText("Denoising...")
        state = {'k': 0, 'n': 0, 'out': None}

        def progress(k, n):
            state['k'], state['n'] = k, n

        def finish(kind, *args):
            state['out'] = (kind, args)

        if not e.denoise_async(
                model,
                lambda r, a: finish('done', r, a),
                lambda ex: finish('error', ex),
                progress):
            self._set_busy(False)
            return
        dlg = wx.ProgressDialog(
            "Denoising", "Loading the model...", maximum=1000, parent=self,
            style=wx.PD_APP_MODAL | wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME |
            wx.PD_REMAINING_TIME | wx.PD_SMOOTH)
        cancelling = False
        while state['out'] is None:
            k, n = state['k'], state['n']
            if cancelling:
                keep = dlg.Pulse("Cancelling...")
            elif n == 0:
                keep = dlg.Pulse("Loading the model...")
            else:
                keep = dlg.Update(min(int(1000 * k / n), 999),
                                  f"Denoising... tile {k} of {n}")
            if isinstance(keep, tuple):
                keep = keep[0]
            if not keep and not cancelling:
                cancelling = True
                e.cancel()
            wx.MilliSleep(50)
            wx.YieldIfNeeded()
        dlg.Destroy()
        kind, args = state['out']
        if kind == 'done':
            self._denoise_done(*args, model)
        elif isinstance(args[0], denoise.Cancelled):
            self._set_busy(False)
            self.statusbar.SetStatusText("Denoising cancelled")
        else:
            self._denoise_failed(args[0])

    def _ensure_model(self, tier):
        """Offer to download the model if its file is missing; return True
        if the model is available."""
        e = self.engine
        if e.options.model_file or tier not in config.DENOISE_MODELS:
            return True
        kind = e.raw.kind
        name, url, size, sha = config.denoise_models(kind)[tier]
        path = e.conf.get_denoise_model_file(tier, kind)
        ok = ensure_model_file(self, f"The '{tier}' denoising model", name,
                               path, url, size, sha)
        if ok is None:
            self.statusbar.SetStatusText("Denoising cancelled: no model")
        elif not ok:
            self.statusbar.SetStatusText("Model download failed")
        return bool(ok)

    def _denoise_done(self, res, after, model):
        self.engine.apply_result(res, after, model)
        self._set_busy(False)
        self.image_panel.show_result()
        iso = "" if not res.iso else \
            f", ISO {res.iso:g}{' assumed' if res.iso_guessed else ''}"
        self.statusbar.SetStatusText(
            f"Denoising done ({res.model} on {res.provider}{iso})")

    def _denoise_failed(self, ex):
        import traceback
        traceback.print_exception(ex)
        self._set_busy(False)
        self.statusbar.SetStatusText("Denoising failed")
        wx.MessageDialog(self, f"Denoising failed:\n{ex}", "Denoising error",
                         wx.OK | wx.ICON_ERROR).ShowModal()

    def on_close(self, event):
        if self.engine.busy_now():
            return
        if not self.check_save():
            return
        def update(cur, prev):
            a, b = prev
            na, nb = cur
            if abs(a - na) >= 5:
                a = na
            if abs(b - nb) >= 5:
                b = nb
            return (a, b)
        self.engine.conf.window_size = \
            update(self.GetSize(), self.engine.conf.window_size)
        if self.mode == 'heal':
            c = self.engine.conf
            c.brush_radius = float(self.engine.radius)
            c.brush_feather = float(self.engine.feather)
            c.brush_opacity = float(self.engine.opacity)
        if self.mode == 'denoise':
            self.engine.conf.denoise_model = \
                self.choice_model.GetStringSelection()
        self.engine.conf.save()
        event.Skip()

# end of class MainFrame


def ensure_model_file(parent, what, name, path, url, size, sha):
    """If the model file `path` is missing, ask whether to download it (with
    a modal progress dialog).  Returns True if the file is available (or has
    just been downloaded), None if the user declined, False if the download
    failed or was aborted."""
    import download
    path = Path(path)
    if path.is_file():
        return True
    mb = f" (about {size / 1e6:.0f} MB)" if size else ""
    dlg = wx.MessageDialog(
        parent, f"{what} ({name}) is not installed.\n\n"
        f"Download it now{mb}?\n\nIt will be saved to:\n{path}",
        "Model not found", wx.YES_NO | wx.YES_DEFAULT | wx.ICON_QUESTION)
    dlg.SetYesNoLabels("Download", "Cancel")
    if dlg.ShowModal() != wx.ID_YES:
        return None
    total = size or 100
    prog = wx.ProgressDialog(
        "Downloading model", f"Downloading {name}...", maximum=1000,
        parent=parent, style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE |
        wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    state = {'n': 0, 'total': total, 'done': False, 'abort': False,
             'error': None}

    def work():
        def progress(n, t):
            state['n'] = n
            state['total'] = t or total
        try:
            download.download(url, path, size, sha, progress,
                              lambda: state['abort'])
        except BaseException as ex:
            state['error'] = ex
        state['done'] = True

    threading.Thread(target=work, daemon=True).start()
    while not state['done']:
        v = min(int(1000 * state['n'] / max(state['total'], 1)), 999)
        keep = prog.Update(
            v, f"Downloading {name}: {state['n'] / 1e6:.0f} of "
               f"{state['total'] / 1e6:.0f} MB")
        if isinstance(keep, tuple):
            keep = keep[0]
        if not keep:
            state['abort'] = True
        wx.MilliSleep(100)
        wx.YieldIfNeeded()
    prog.Destroy()
    if state['error'] is not None:
        if not state['abort']:
            wx.MessageDialog(
                parent, f"Download failed:\n{state['error']}",
                "Download error", wx.OK | wx.ICON_ERROR).ShowModal()
        return False
    return True


def ensure_mode_model(conf, mode):
    """Before starting: offer to download the model needed by `mode` (mask
    or heal) if it is missing.  Returns False if the program should not go on."""
    if mode == 'mask':
        found = [m for m in config.MASK_MODELS.values()
                 if m[0] == conf.model]
        if not found:        # a custom model: nothing we can download
            return True
        name, url, size, sha = found[0]
        what = "The SAM2 masking model"
        path = conf.get_model_file()
    elif mode == 'heal':
        import lama
        backend = lama.resolve_backend_name(conf)
        name, url, size, sha = config.HEAL_MODELS[backend]
        what = "The LaMa healing model"
        path = conf.get_heal_model_file(backend)
    else:
        return True
    return ensure_model_file(None, what, name, path, url, size, sha) is True


def app_icon():
    """The application icon, or None if it cannot be loaded."""
    icon_path = config.resource_dir() / 'icons' / 'app-icon.png'
    if not icon_path.exists():
        return None
    icon = wx.Icon(str(icon_path), wx.BITMAP_TYPE_PNG)
    return icon if icon.IsOk() else None


_dock_icon = None   # keep a reference, or the icon is garbage collected


def set_dock_icon():
    """macOS: replace the generic Python icon in the Dock (when run as a
    script, rather than from an application bundle) with ours."""
    global _dock_icon
    icon = app_icon()
    if wx.Platform != '__WXMAC__' or icon is None:
        return
    try:
        from wx.adv import TaskBarIcon, TBI_DOCK
        _dock_icon = TaskBarIcon(TBI_DOCK)
        _dock_icon.SetIcon(icon)
    except Exception:
        pass


def ask_mode():
    """Ask which mode to use; return None if the dialog is cancelled."""
    modes = [('mask', "AI masking: build a mask from clicks"),
             ('heal', "AI healing: remove objects (generative fill)"),
             ('denoise', "Raw denoising of a mosaiced DNG")]
    # (wx.SingleChoiceDialog has a broken layout on macOS: use plain sizers)
    dlg = wx.Dialog(None, title="SMART")
    icon = app_icon()
    if icon is not None:
        dlg.SetIcon(icon)
    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(wx.StaticText(dlg, label="Select the mode to use:"),
              0, wx.ALL, 12)
    radios = []
    for i, (_, label) in enumerate(modes):
        r = wx.RadioButton(dlg, label=label,
                           style=wx.RB_GROUP if i == 0 else 0)
        radios.append(r)
        sizer.Add(r, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
    radios[0].SetValue(True)
    btns = dlg.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
    sizer.Add(btns, 0, wx.EXPAND | wx.ALL, 12)
    dlg.SetSizerAndFit(sizer)
    dlg.CentreOnScreen()
    try:
        if dlg.ShowModal() != wx.ID_OK:
            return None
        return next(m for (m, _), r in zip(modes, radios) if r.GetValue())
    finally:
        dlg.Destroy()


def main(conf, filename=None, mode=None, outdir=None, opts=None):
    app = wx.PyApp() if wx.Platform == '__WXMAC__' else wx.App()
    set_dock_icon()
    fn = config.Config.get_config_file()
    if not os.path.exists(fn):
        conf.save()
        wx.MessageDialog(None, f"Configuration file created in:\n{fn}\n"
                         "You can adjust the settings to your "
                         "environment later.",
                         "Initial configuration", wx.OK).ShowModal()
    if mode is None:
        mode = ask_mode()
        if mode is None:
            return
    try:
        if not ensure_mode_model(conf, mode):
            return
        if mode == 'heal':
            import heal_engine
            eng = heal_engine.AIHealingEngine(conf)
        elif mode == 'denoise':
            import denoise_engine
            eng = denoise_engine.DenoiseEngine(conf, opts)
        else:
            import engine
            eng = engine.AIMaskingEngine(conf)
        frame = MainFrame(conf, eng, mode, outdir,
                          getattr(opts, 'save_as', None))
        def fixgeom():
            x, y = frame.GetPosition()
            w, h = frame.GetSize()
            r = wx.GetClientDisplayRect()
            w = min(w, r.GetWidth())
            h = min(h, r.GetHeight())
            x = max(x, r.GetX())
            y = max(y, r.GetY())
            frame.SetPosition((x, y))
            frame.SetSize((w, h))
            frame.Raise()
            frame.Show()
        wx.CallAfter(fixgeom)
        if filename is not None:
            wx.CallAfter(lambda : frame.load_image(filename))
        app.MainLoop()
    except Exception as e:
        import traceback
        traceback.print_exc()
        wx.MessageDialog(None, f"Fatal error:\n{e}\n"
                         f"Please check your configuration file in {fn}",
                         "Error",
                         wx.OK | wx.ICON_ERROR).ShowModal()
