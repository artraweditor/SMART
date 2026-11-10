"""Preferences dialog: edits the settings stored in the configuration file."""

import dataclasses
import wx

import config


def _providers():
    try:
        import onnxruntime
        avail = [p.replace('ExecutionProvider', '').lower()
                 for p in onnxruntime.get_available_providers()]
    except Exception:
        avail = []
    res = ['auto', 'cpu']
    res += [p for p in avail if p not in res]
    return res


# Field specs: (attribute, label, kind, extra, tooltip)
# kinds: text, file, dir, colour, int, float, bool, choice, combo, size,
# providers (comma-separated list, empty = None)
# 'provider' and 'dng_bits' are not config keys: they are common settings
# that are applied to the corresponding keys of all the modes (see
# PreferencesDialog.expand).
def _tabs():
    provs = _providers()
    return [
        ('Common', 'common', [
            ('provider', "Execution provider", 'combo', provs,
             "Where the neural networks run (onnxruntime provider), for all "
             "modes: auto picks the best available (GPU/CoreML if present), "
             "cpu always works. Changing it overrides the per-mode values "
             "in the config file."),
            ('dng_bits', "DNG bits", 'choice', [16, 32],
             "Bit depth of the DNG files written by denoise and upscale: "
             "16 = integer (smaller files), 32 = floating point (larger, "
             "no precision loss)."),
            ('exiftool', "ExifTool", 'file', None,
             "Path of the exiftool executable, used to copy the metadata "
             "of the input image to the output."),
            ('display_icc_profile', "Display ICC profile", 'file', None,
             "ICC profile of your monitor, to show the image with correct "
             "colours (empty: none). Applied at the next start."),
            ('background_color', "Background colour", 'colour', None,
             "Colour around the image in the preview."),
            ('window_size', "Window size", 'size', None,
             "Size of the main window, in pixels. Applied at the next "
             "start (it is updated automatically when you resize and quit)."),
            ('last_dir', "Last used directory", 'dir', None,
             "Directory where the open/save dialogs start."),
        ]),
        ('Mask', 'mask', [
            ('model', "Model quality", 'choice',
             list(config.MASK_MODELS),
             "SAM2.1 network used to build masks: tiny is fastest, large "
             "is the most accurate but slow and needs a big download. "
             "Applied at the next start."),
            ('mask_color', "Mask colour", 'colour', None,
             "Colour used to overlay the mask on the image."),
        ]),
        ('Heal', 'heal', [
            ('heal_max_tile', "Max tile size", 'int', (64, 16384),
             "Largest window (in pixels) sent to the network at once. "
             "Bigger gives more context but uses more memory and time."),
            ('heal_seam_match', "Match seams", 'bool', None,
             "Blend the edges of the healed area so that no seam is "
             "visible."),
            ('heal_even', "Even out tones", 'bool', None,
             "Correct brightness/colour differences between the healed "
             "area and its surroundings."),
            ('heal_regrain', "Re-add grain", 'bool', None,
             "Add noise to the healed area to match the grain of the "
             "rest of the image."),
            ('brush_radius', "Brush radius", 'float', (1, 2000, 1, 0),
             "Default radius of the painting brush, in pixels."),
            ('brush_feather', "Brush feather", 'float', (0, 1, 0.05, 2),
             "Default softness of the brush edge (0 = hard, 1 = fully "
             "soft)."),
            ('brush_opacity', "Brush opacity", 'float', (0.01, 1, 0.05, 2),
             "Default opacity of the brush strokes."),
            ('brush_color', "Brush colour", 'colour', None,
             "Colour used to show the painted area."),
        ]),
        ('Denoise', 'denoise', [
            ('denoise_model', "Model", 'choice',
             list(config.DENOISE_MODELS),
             "Denoising network: small is fastest, large gives the "
             "best quality."),
            ('denoise_tile', "Tile size", 'int', (64, 16384),
             "The image is processed in square tiles of this size (in "
             "pixels): smaller tiles use less memory."),
            ('denoise_margin', "Tile margin", 'int', (0, 1024),
             "Extra pixels processed around each tile to avoid visible "
             "tile borders. Must be less than half of the tile size."),
        ]),
        ('Upscale', 'upscale', [
            ('upscale_scale', "Scale", 'choice',
             list(config.UPSCALE_MODELS),
             "Enlargement factor: 2x or 4x."),
            ('upscale_tile', "Tile size", 'int', (64, 16384),
             "The image is processed in square tiles of this size (in "
             "input pixels): smaller tiles use less memory."),
            ('upscale_margin', "Tile margin", 'int', (0, 1024),
             "Extra pixels processed around each tile to avoid visible "
             "tile borders. Must be less than half of the tile size."),
            ('upscale_compare', "Compare with", 'choice',
             ['nearest', 'lanczos'],
             "How the input is enlarged in the \"before\" view: nearest "
             "neighbour (sharp pixels) or Lanczos (smooth)."),
        ]),
    ]


class _ColourCtrl(wx.BitmapButton):
    """A button showing a colour; click to change it.  (wx.ColourPickerCtrl
    crashes on macOS when the app is a wx.PyApp, as in SMART.)"""
    def __init__(self, parent):
        super().__init__(parent, bitmap=self._bitmap((0, 0, 0)))
        self.colour = (0, 0, 0)
        self.Bind(wx.EVT_BUTTON, self.on_click)

    @staticmethod
    def _bitmap(rgb):
        bmp = wx.Bitmap(48, 18)
        dc = wx.MemoryDC(bmp)
        dc.SetBackground(wx.Brush(wx.Colour(*rgb)))
        dc.Clear()
        dc.SetPen(wx.Pen(wx.Colour(90, 90, 90)))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(0, 0, 48, 18)
        dc.SelectObject(wx.NullBitmap)
        return bmp

    def SetColour(self, rgb):
        self.colour = tuple(rgb)
        self.SetBitmapLabel(self._bitmap(self.colour))
        self.Refresh()

    def GetColour(self):
        return self.colour

    def on_click(self, event):
        data = wx.ColourData()
        data.SetColour(wx.Colour(*self.colour))
        dlg = wx.ColourDialog(self, data)
        if dlg.ShowModal() == wx.ID_OK:
            c = dlg.GetColourData().GetColour()
            self.SetColour((c.Red(), c.Green(), c.Blue()))
        dlg.Destroy()


class _PathCtrl(wx.Panel):
    """A text field with a Browse button."""
    def __init__(self, parent, is_dir):
        super().__init__(parent)
        self.is_dir = is_dir
        self.text = wx.TextCtrl(self)
        btn = wx.Button(self, label="Browse...")
        btn.Bind(wx.EVT_BUTTON, self.on_browse)
        s = wx.BoxSizer(wx.HORIZONTAL)
        s.Add(self.text, 1, wx.ALIGN_CENTER_VERTICAL)
        s.Add(btn, 0, wx.LEFT, 6)
        self.SetSizer(s)

    def on_browse(self, event):
        if self.is_dir:
            dlg = wx.DirDialog(self, "Choose a directory",
                               self.text.GetValue())
        else:
            dlg = wx.FileDialog(self, "Choose a file",
                                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.text.SetValue(dlg.GetPath())
        dlg.Destroy()


_COMMON = ('provider', 'dng_bits')


class PreferencesDialog(wx.Dialog):
    def __init__(self, parent, conf, mode=None):
        super().__init__(parent, title="Preferences",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.conf = conf
        self.widgets = {}   # attr -> (kind, extra, widget(s))
        nb = wx.Notebook(self)
        sel = 0
        for i, (title, key, fields) in enumerate(_tabs()):
            page = wx.ScrolledWindow(nb)
            page.SetScrollRate(0, 10)
            grid = wx.FlexGridSizer(cols=2, vgap=8, hgap=10)
            grid.AddGrowableCol(1)
            for attr, label, kind, extra, tip in fields:
                w = self._make(page, kind, extra)
                if tip:
                    w.SetToolTip(tip)
                self.widgets[attr] = (kind, extra, w)
                if kind == 'bool':
                    grid.AddSpacer(0)
                    w.SetLabel(label)
                    grid.Add(w, 0, wx.ALIGN_CENTER_VERTICAL)
                else:
                    lbl = wx.StaticText(page, label=label)
                    if tip:
                        lbl.SetToolTip(tip)
                    grid.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL)
                    grid.Add(w, 1, wx.EXPAND)
            border = wx.BoxSizer(wx.VERTICAL)
            border.Add(grid, 1, wx.EXPAND | wx.ALL, 12)
            page.SetSizer(border)
            nb.AddPage(page, title)
            if key == mode:
                sel = i
        nb.SetSelection(sel)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(nb, 1, wx.EXPAND | wx.ALL, 8)
        btns = wx.BoxSizer(wx.HORIZONTAL)
        defaults = wx.Button(self, label="Restore defaults")
        defaults.Bind(wx.EVT_BUTTON, self.on_defaults)
        btns.Add(defaults, 0, wx.RIGHT, 8)
        btns.AddStretchSpacer()
        std = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        btns.Add(std, 0)
        vbox.Add(btns, 0, wx.EXPAND | wx.ALL, 8)
        self.SetSizerAndFit(vbox)
        self.SetMinSize(wx.Size(520, 360))
        self.SetSize(wx.Size(560, 420))
        self.Bind(wx.EVT_BUTTON, self.on_ok, id=wx.ID_OK)
        self.load(conf)
        # the common settings stand for several keys: only overwrite those
        # if the user changes them
        self._initial = {k: self._get(k) for k in _COMMON}

    # ------------------------------------------------------------ widgets
    def _make(self, parent, kind, extra):
        if kind == 'text':
            return wx.TextCtrl(parent)
        if kind in ('file', 'dir'):
            return _PathCtrl(parent, kind == 'dir')
        if kind == 'providers':
            return wx.TextCtrl(parent)
        if kind == 'colour':
            return _ColourCtrl(parent)
        if kind == 'bool':
            return wx.CheckBox(parent)
        if kind == 'int':
            return wx.SpinCtrl(parent, min=extra[0], max=extra[1])
        if kind == 'float':
            lo, hi, inc, digits = extra
            return wx.SpinCtrlDouble(parent, min=lo, max=hi, inc=inc,
                                     initial=lo)
        if kind == 'choice':
            return wx.Choice(parent, choices=[str(x) for x in extra])
        if kind == 'combo':
            return wx.ComboBox(parent, choices=extra)
        if kind == 'size':
            p = wx.Panel(parent)
            s = wx.BoxSizer(wx.HORIZONTAL)
            p.w = wx.SpinCtrl(p, min=200, max=20000)
            p.h = wx.SpinCtrl(p, min=200, max=20000)
            s.Add(p.w)
            s.Add(wx.StaticText(p, label=" x "), 0, wx.ALIGN_CENTER_VERTICAL)
            s.Add(p.h)
            p.SetSizer(s)
            return p
        raise ValueError(kind)

    def _set(self, attr, value):
        kind, extra, w = self.widgets[attr]
        if kind == 'text':
            w.SetValue(value or '')
        elif kind in ('file', 'dir'):
            w.text.SetValue(value or '')
        elif kind == 'providers':
            w.SetValue(', '.join(value) if value else '')
        elif kind == 'colour':
            w.SetColour(tuple(int(c) for c in value))
        elif kind == 'bool':
            w.SetValue(bool(value))
        elif kind == 'int':
            w.SetValue(int(value))
        elif kind == 'float':
            w.SetDigits(extra[3])
            w.SetValue(float(value))
        elif kind == 'choice':
            names = [str(x) for x in extra]
            if str(value) in names:
                w.SetSelection(names.index(str(value)))
        elif kind == 'combo':
            w.SetValue(str(value))
        elif kind == 'size':
            w.w.SetValue(int(value[0]))
            w.h.SetValue(int(value[1]))

    def _get(self, attr):
        kind, extra, w = self.widgets[attr]
        if kind == 'text':
            return w.GetValue().strip()
        if kind in ('file', 'dir'):
            return w.text.GetValue().strip() or (
                None if attr == 'display_icc_profile' else '')
        if kind == 'providers':
            v = [p.strip() for p in w.GetValue().split(',') if p.strip()]
            return v or None
        if kind == 'colour':
            return tuple(w.GetColour())
        if kind in ('bool', 'int', 'float'):
            return w.GetValue()
        if kind == 'choice':
            return extra[w.GetSelection()]
        if kind == 'combo':
            return w.GetValue().strip()
        if kind == 'size':
            return (w.w.GetValue(), w.h.GetValue())

    # ------------------------------------------------------------ actions
    def load(self, conf):
        for attr in self.widgets:
            if attr == 'provider':
                self._set(attr, conf.denoise_provider)
            elif attr == 'dng_bits':
                self._set(attr, conf.denoise_dng_bits)
            else:
                self._set(attr, getattr(conf, attr))

    def on_defaults(self, event):
        self.load(config.Config())

    def values(self):
        return {attr: self._get(attr) for attr in self.widgets}

    def expand(self, vals):
        """Replace the common settings with the config keys they stand for."""
        p = vals.pop('provider') or 'auto'
        bits = vals.pop('dng_bits')
        if p != self._initial['provider']:
            vals['mask_provider'] = p
            vals['heal_providers'] = None if p == 'auto' else [p]
            vals['denoise_provider'] = vals['upscale_provider'] = p
        if bits != self._initial['dng_bits']:
            vals['denoise_dng_bits'] = vals['upscale_dng_bits'] = bits
        return vals

    def on_ok(self, event):
        vals = self.values()
        for p in ('denoise', 'upscale'):
            if vals[p + '_tile'] <= 2 * vals[p + '_margin']:
                wx.MessageDialog(
                    self, f"The {p} tile size must be larger than twice "
                    "the margin.", "Invalid value",
                    wx.OK | wx.ICON_ERROR).ShowModal()
                return
        self.result = self.expand(vals)
        event.Skip()

    def apply(self, conf):
        """Store the edited values into `conf` (and save it); returns the
        names of the attributes that changed."""
        changed = set()
        for attr, v in self.result.items():
            old = getattr(conf, attr)
            if isinstance(old, list):
                old = tuple(old)
            if old != (tuple(v) if isinstance(v, list) else v):
                changed.add(attr)
                setattr(conf, attr, v)
        conf.save()
        return changed
