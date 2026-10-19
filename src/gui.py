import wx
import os
from pathlib import Path
from PIL import Image

import config
import engine

class ImagePanel(wx.Panel):
    def __init__(self, parent, engine):
        super().__init__(parent)
        self.image = None
        self.bitmap = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.drag_start = (0, 0)
        self.engine = engine

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        self.Bind(wx.EVT_MOTION, self.on_motion)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mousewheel)
        self.SetBackgroundColour(
            wx.Colour(*self.engine.conf.background_color))

    def load_image(self, path):
        self.engine.load_image(path)
        self.image = self.engine.displayed_image
        self.update_bitmap()
        self.center_image()
        self.zoom_fit()

    def update_bitmap(self):
        if self.image is not None:
            h, w = self.image.shape[:2]
            wx_image = wx.Image(w, h)
            wx_image.SetData(self.image.tobytes())
            self.bitmap = wx.Bitmap(wx_image)
        else:
            self.bitmap = None

    def center_image(self):
        if self.bitmap:
            panel_w, panel_h = self.GetSize()
            img_w = int(self.bitmap.GetWidth() * self.zoom)
            img_h = int(self.bitmap.GetHeight() * self.zoom)
            self.offset_x = (panel_w - img_w) // 2
            self.offset_y = (panel_h - img_h) // 2

    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self)
        dc.Clear()
        if self.bitmap:
            w = int(self.bitmap.GetWidth() * self.zoom)
            h = int(self.bitmap.GetHeight() * self.zoom)
            scaled = self.bitmap.ConvertToImage().Scale(w, h,
                                                        wx.IMAGE_QUALITY_HIGH)
            bmp = wx.Bitmap(scaled)
            x = int(self.offset_x)
            y = int(self.offset_y)
            dc.DrawBitmap(bmp, x, y, True)

            dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), width=1))
            radius = 4
            colors = [wx.Colour(255, 0, 0), wx.Colour(0, 255, 0)]
            for p, b in zip(self.engine.points, self.engine.labels):
                sx, sy = self.to_screen_coords(p[0], p[1])
                dc.SetBrush(wx.Brush(colors[b], wx.BRUSHSTYLE_SOLID))
                dc.DrawCircle(sx, sy, radius)

    def _panning(self):
        return not wx.GetKeyState(wx.WXK_SHIFT)

    def on_left_down(self, event):
        if self.image is not None:
            if self._panning():
                self.dragging = True
                self.drag_start = (event.GetX(), event.GetY())
                self.CaptureMouse()
            else:
                self.engine.add_point(
                    self.to_image_coords(event.GetX(), event.GetY()), True)
                self.image = self.engine.displayed_image
                self.update_bitmap()
                self.Refresh()

    def on_left_up(self, event):
        if self.image is not None and self.dragging:
            self.dragging = False
            #self.center_image()
            if self.HasCapture():
                self.ReleaseMouse()
            self.Refresh()

    def on_right_down(self, event):
        if self.image is not None and not self._panning():
            self.engine.add_point(
                self.to_image_coords(event.GetX(), event.GetY()), False)
            self.image = self.engine.displayed_image
            self.update_bitmap()
            self.Refresh()
            
    def on_motion(self, event):
        if self.dragging and event.Dragging() and event.LeftIsDown():
            dx = event.GetX() - self.drag_start[0]
            dy = event.GetY() - self.drag_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.drag_start = (event.GetX(), event.GetY())
            self.Refresh()

    def on_mousewheel(self, event):
        rotation = event.GetWheelRotation()
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
        if self.bitmap:
            panel_w, panel_h = self.GetSize()
            img_w, img_h = self.bitmap.GetWidth(), self.bitmap.GetHeight()
            scale_w = panel_w / img_w
            scale_h = panel_h / img_h
            self.zoom = min(scale_w, scale_h)
            self.offset_x = (panel_w - img_w * self.zoom) // 2
            self.offset_y = (panel_h - img_h * self.zoom) // 2
            self.Refresh()

    def reset(self, clear_image=False):
        self.engine.reset(clear_image)
        self.image = self.engine.displayed_image
        self.update_bitmap()
        self.Refresh()

    def undo(self):
        self.engine.undo_last()
        self.image = self.engine.displayed_image
        self.update_bitmap()
        self.Refresh()        

    def to_image_coords(self, x, y):
        w = self.bitmap.GetWidth() * self.zoom
        h = self.bitmap.GetHeight() * self.zoom
        real_w, real_h = self.engine.get_size()
        scale_w = real_w / w
        scale_h = real_h / h
        return (int(round((x - self.offset_x) * scale_w)),
                int(round((y - self.offset_y) * scale_h)))

    def to_screen_coords(self, x, y):
        w = self.bitmap.GetWidth() * self.zoom
        h = self.bitmap.GetHeight() * self.zoom
        real_w, real_h = self.engine.get_size()
        scale_w = real_w / w
        scale_h = real_h / h
        return (int(round(x / scale_w + self.offset_x)),
                int(round(y / scale_h + self.offset_y)))

# end of class ImagePanel


class MainFrame(wx.Frame):
    def __init__(self, conf, engine):
        super().__init__(None, title="SMART AI mask builder - v" +
                         config.version,
                         size=conf.window_size)

        icon_path = \
            Path(__file__).resolve().parent.parent / 'icons' / 'app-icon.png'
        if icon_path.exists():
            icon = wx.Icon(str(icon_path), wx.BITMAP_TYPE_PNG)
            self.SetIcon(icon)

        self.engine = engine
        self.filename = None

        self.statusbar = self.CreateStatusBar()
        msg = "shift+left click: add positive point; " \
            "shift+right click: add negative point   "
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
        bg = toolbar.GetBackgroundColour()
        l = bg[0]/255.0 * 0.213 + bg[1]/255.0 * 0.715 + bg[2]/255.0 * 0.072
        repl = b'#ffffff' if l < 0.5 else b'#000000'
        d = Path(__file__).resolve().parent.parent / 'icons'
        sz = wx.Size(32, 32)
        def svg(pth):
            with open(d / pth, 'rb') as f:
                data = f.read().replace(b'#2a7fff', repl)
                return wx.BitmapBundle.FromSVG(data, sz)

        tb_open = toolbar.AddTool(
            wx.ID_OPEN, "Open", svg('folder-open.svg'),
            shortHelp="Open image...")
        tb_save = toolbar.AddTool(
            wx.ID_SAVE, "Save", svg('save.svg'),
            shortHelp="Save mask...")
        tb_reset = toolbar.AddTool(
            wx.ID_ANY, "Reset", svg('undo-all.svg'),
            shortHelp="Remove all points")
        tb_undo = toolbar.AddTool(
            wx.ID_ANY, "Undo", svg('undo.svg'),
            shortHelp="Remove last added point")
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

        self.Bind(wx.EVT_MENU, self.on_close, id=wx.ID_EXIT)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.image_panel = ImagePanel(self, engine)
        hbox.Add(self.image_panel, proportion=1, flag=wx.EXPAND)

        vbox.Add(hbox, proportion=1, flag=wx.EXPAND)
        self.SetSizer(vbox)
        self.Raise()

    def check_save(self):
        if self.engine.saved:
            return True
        with wx.MessageDialog(self,
                              "You have unsaved changes. "
                              "Do you want to save the mask before continuing?",
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
            self.statusbar.SetStatusText("No mask to save")
            return True
        dd = str(self.filename.parent)
        df = str(self.filename.stem + "_mask.png")
        # NOTE: wxWidgets 3.2.9 on macOS asserts (index out of bounds) in
        # FileDialog.ShowModal() with FD_SAVE and a single wildcard filter,
        # so we add a second one as a workaround
        with wx.FileDialog(self, "Save Mask",
                           wildcard="PNG files (*.png)|*.png|"
                           "All files (*.*)|*.*",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                           defaultDir=dd, defaultFile=df) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return True
            path = fd.GetPath()
            try:
                self.engine.save_mask(path)
                self.statusbar.SetStatusText(f"mask saved to {path}")
                return True
            except Exception as e:
                wx.MessageDialog(self, f"Error saving mask:\n{e}",
                                 "Save Error",
                                 wx.OK | wx.ICON_ERROR).ShowModal()
                return False

    def on_save(self, event):
        self.save_mask()

    def on_reset(self, event):
        self.image_panel.reset()

    def on_undo(self, event):
        self.image_panel.undo()

    def on_close(self, event):
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
        self.engine.conf.save()
        event.Skip()

# end of class MainFrame


def main(conf, filename=None):
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
            frame = MainFrame(conf, engine.AIMaskingEngine(conf))
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
