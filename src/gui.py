import wx
import os
from pathlib import Path
import numpy as np
from PIL import Image

import config

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


class MainFrame(wx.Frame):
    def __init__(self, conf, engine, mode='mask', outdir=None):
        self.mode = mode
        self.outdir = outdir    # default directory of the save dialog
        title = "SMART AI mask builder" if mode == 'mask' \
            else "SMART AI healing"
        super().__init__(None, title=title + " - v" + config.version,
                         size=conf.window_size)

        icon_path = \
            Path(__file__).resolve().parent.parent / 'icons' / 'app-icon.png'
        if icon_path.exists():
            icon = wx.Icon(str(icon_path), wx.BITMAP_TYPE_PNG)
            self.SetIcon(icon)

        self.engine = engine
        self.filename = None
        self.busy_cursor = None

        self.statusbar = self.CreateStatusBar()
        if mode == 'mask':
            msg = "shift+left click: add positive point; " \
                "shift+right click: add negative point   "
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
        bg = toolbar.GetBackgroundColour()
        l = bg[0]/255.0 * 0.213 + bg[1]/255.0 * 0.715 + bg[2]/255.0 * 0.072
        repl = b'#ffffff' if l < 0.5 else b'#000000'
        d = Path(__file__).resolve().parent.parent / 'icons'
        sz = wx.Size(32, 32)
        def svg(pth):
            with open(d / pth, 'rb') as f:
                data = f.read().replace(b'#2a7fff', repl)
                return wx.BitmapBundle.FromSVG(data, sz)

        what = "mask" if mode == 'mask' else "image"
        tb_open = toolbar.AddTool(
            wx.ID_OPEN, "Open", svg('folder-open.svg'),
            shortHelp="Open image...")
        tb_save = toolbar.AddTool(
            wx.ID_SAVE, "Save", svg('save.svg'),
            shortHelp=f"Save {what}...")
        tb_reset = toolbar.AddTool(
            wx.ID_ANY, "Reset", svg('undo-all.svg'),
            shortHelp="Remove all points" if mode == 'mask'
            else "Revert to the image as loaded")
        tb_undo = toolbar.AddTool(
            wx.ID_ANY, "Undo", svg('undo.svg'),
            shortHelp="Remove last added point" if mode == 'mask'
            else "Undo last stroke or healing")
        tb_zoom_in = toolbar.AddTool(
            wx.ID_ZOOM_IN, "Zoom In", svg('magnifier-plus.svg'),
            shortHelp="Zoom in")
        tb_zoom_out = toolbar.AddTool(
            wx.ID_ZOOM_OUT, "Zoom Out", svg('magnifier-minus.svg'),
            shortHelp="Zoom out")
        tb_zoom_1_1 = toolbar.AddTool(
            wx.ID_ANY, "Zoom 1:1", svg('magnifier-1to1.svg'),
            shortHelp="Zoom to 100%")
        tb_zoom_fit = toolbar.AddTool(
            wx.ID_ANY, "Zoom Fit", svg('magnifier-fit.svg'),
            shortHelp="Zoom to fit")
        tb_heal = tb_clear = None
        heal_key = "Cmd+Enter" if wx.Platform == '__WXMAC__' else "Ctrl+Enter"
        if mode == 'heal':
            toolbar.AddSeparator()
            tb_heal = toolbar.AddTool(
                wx.ID_ANY, "Heal", svg('heal.svg'),
                shortHelp=f"Heal the painted areas ({heal_key})")
            tb_clear = toolbar.AddTool(
                wx.ID_ANY, "Clear strokes", svg('brush-clear.svg'),
                shortHelp="Remove the painted areas (Esc)")
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
        toolbar.Realize()

        accel_entries = [
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('o'), tb_open.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('s'), tb_save.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('z'), tb_undo.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('r'), tb_reset.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('+'), tb_zoom_in.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('='), tb_zoom_in.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('-'), tb_zoom_out.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('1'), tb_zoom_1_1.GetId()),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('f'), tb_zoom_fit.GetId()),
        ]
        if mode == 'heal':
            accel_entries += [
                wx.AcceleratorEntry(wx.ACCEL_CTRL, wx.WXK_RETURN,
                                    tb_heal.GetId()),
                wx.AcceleratorEntry(wx.ACCEL_CTRL, wx.WXK_NUMPAD_ENTER,
                                    tb_heal.GetId()),
                wx.AcceleratorEntry(wx.ACCEL_NORMAL, wx.WXK_ESCAPE,
                                    tb_clear.GetId()),
            ]
        accel_table = wx.AcceleratorTable(accel_entries)
        self.SetAcceleratorTable(accel_table)

        self.Bind(wx.EVT_MENU, self.on_open_image, tb_open)
        self.Bind(wx.EVT_MENU, self.on_save, tb_save)
        self.Bind(wx.EVT_MENU, self.on_reset, tb_reset)
        self.Bind(wx.EVT_MENU, self.on_undo, tb_undo)
        self.Bind(wx.EVT_MENU, self.panel_zoom_in, tb_zoom_in)
        self.Bind(wx.EVT_MENU, self.panel_zoom_out, tb_zoom_out)
        self.Bind(wx.EVT_MENU, self.panel_zoom_1_1, tb_zoom_1_1)
        self.Bind(wx.EVT_MENU, self.panel_zoom_fit, tb_zoom_fit)

        self.Bind(wx.EVT_TOOL, self.on_open_image, tb_open)
        self.Bind(wx.EVT_TOOL, self.panel_zoom_in, tb_zoom_in)
        self.Bind(wx.EVT_TOOL, self.panel_zoom_out, tb_zoom_out)
        self.Bind(wx.EVT_TOOL, self.panel_zoom_1_1, tb_zoom_1_1)
        self.Bind(wx.EVT_TOOL, self.panel_zoom_fit, tb_zoom_fit)
        self.Bind(wx.EVT_TOOL, self.on_save, tb_save)
        self.Bind(wx.EVT_TOOL, self.on_reset, tb_reset)
        self.Bind(wx.EVT_TOOL, self.on_undo, tb_undo)
        if mode == 'heal':
            for ev in (wx.EVT_MENU, wx.EVT_TOOL):
                self.Bind(ev, self.on_heal, tb_heal)
                self.Bind(ev, self.on_clear_strokes, tb_clear)

        self.Bind(wx.EVT_MENU, self.on_close, id=wx.ID_EXIT)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.image_panel = ImagePanel(self, engine, mode)
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
        with wx.FileDialog(
                self, "Open Image file",
                wildcard="Image files (*.png;*.jpg;*.jpeg;*.tif;*.tiff)" \
                "|*.png;*.jpg;*.jpeg;*.tif;*.tiff",
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
        dd = self.outdir or str(self.filename.parent)
        if self.mode == 'mask':
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
        with wx.FileDialog(self, title,
                           wildcard=wildcard,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                           defaultDir=dd, defaultFile=df) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return True
            path = fd.GetPath()
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
    def on_clear_strokes(self, event):
        box = self.engine.clear_strokes()
        if box is not None:
            self.image_panel.refresh_image_box(box)

    def _set_busy(self, busy):
        self.image_panel.input_enabled = not busy
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
        e = self.engine
        if e.img is None:
            return
        if not e.has_cover():
            self.statusbar.SetStatusText(
                "Nothing to heal: paint over the image with shift+drag")
            return
        self._set_busy(True)
        self.statusbar.SetStatusText(
            "Healing... (the first run loads the model)")

        def progress(k, n):
            wx.CallAfter(self.statusbar.SetStatusText,
                         f"Healing... area {min(k + 1, n)} of {n}")

        if not e.heal_async(
                lambda p, c: wx.CallAfter(self._heal_done, p, c),
                lambda ex: wx.CallAfter(self._heal_failed, ex),
                progress):
            self._set_busy(False)

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
        self.engine.conf.save()
        event.Skip()

# end of class MainFrame


def main(conf, filename=None, mode='mask', outdir=None):
    app = wx.PyApp() if wx.Platform == '__WXMAC__' else wx.App()
    fn = config.Config.get_config_file()
    if not os.path.exists(fn):
        conf.save()
        wx.MessageDialog(None, f"Configuration file created in:\n{fn}\n"
                         "Please adjust the settings to your "
                         "environment and run again",
                         "Initial configuration", wx.OK).ShowModal()
    else:
        try:
            if mode == 'heal':
                import heal_engine
                eng = heal_engine.AIHealingEngine(conf)
            else:
                import engine
                eng = engine.AIMaskingEngine(conf)
            frame = MainFrame(conf, eng, mode, outdir)
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
