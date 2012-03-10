"""
Fancy image viewer for Matplotlib backend.

TODO:
    - Add interface (menu?) for connecting a Widget to an ImageWindow.
    - Add left-/right-arrow events for scrolling through image collection.
    - Add support for RGB images
    - Add SliderWidget base class

.. note::
    The Qt4Agg doesn't call the close_event correctly [1]_, and so the widget
    will not get cleaned up.

.. [1] https://github.com/matplotlib/matplotlib/pull/716
"""
import numpy as np
import scipy.ndimage as ndi
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.util.dtype import dtype_range


__all__ = ['mplviewer', 'ImageWindow', 'Widget', 'LineProfile']


class ImageWindow(object):
    """Window for displaying images.

    This window is a simple container object that holds a Matplotlib axes
    for showing images. This doesn't subclass the Matplotlib axes (or figure)
    because there be dragons.
    """

    def __init__(self, image, collection=False, **kwargs):
        self._image = image.copy()
        self.fig, self.ax = figimage(image, **kwargs)
        self.ax.autoscale(enable=False)
        self.canvas = self.fig.canvas

        if len(self.ax.images) > 0:
            self._img = self.ax.images[-1].get_array()
        else:
            raise ValueError("No image found in figure")

        self.ax.format_coord = self._format_coord

        if collection:
            pass
            # add left and right arrow key press events to cycle images.

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        self._image = image
        self.ax.images[0].set_array(image)

    def connect_event(self, event, callback):
        cid = self.canvas.mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id):
        self.canvas.mpl_disconnect(callback_id)

    def add_artist(self, artist):
        self.ax.add_artist(artist)

    def remove_artist(self, artist):
        """Disconnect all artists created by this widget."""
        #self.ax.remove(artist)
        self.ax.lines = []

    def redraw(self):
        self.canvas.draw_idle()

    def _format_coord(self, x, y):
        # callback function to format coordinate display in toolbar
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%s @ [%4i, %4i]" % (self.image[y, x], x, y)
        except IndexError:
            return ""


def figimage(img, scale=1, dpi=None, **kwargs):
    """Return figure and axes with figure tightly surrounding image.

    Unlike pyplot.figimage, this actually plots onto an axes object, which
    fills the figure. Plotting the image onto an axes allows for subsequent
    overlays.

    Parameters
    ----------
    img : array
        image to plot
    scale : float
        If scale is 1, the figure and axes have the same dimension as the
        image.  Smaller values of `scale` will shrink the figure.
    dpi : int
        Dots per inch for figure. If None, use the default rcParam.
    """
    dpi = dpi if dpi is not None else plt.rcParams['figure.dpi']

    h, w = img.shape
    figsize = np.array((w, h), dtype=float) / dpi * scale

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    ax.set_axis_off()
    ax.imshow(img, **kwargs)
    return fig, ax


class Widget(object):
    """Base class for widgets that interact with the axes.

    Parameters
    ----------
    image_window : ImageWindow instance.
        Window containing image used in measurement/manipulation.
    useblit : bool
        If True, use blitting to speed up animation. Only available on some
        backends. If None, set to True when using Agg backend, otherwise False.

    Attributes
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        The parent axes for the widget
    canvas : :class:`~matplotlib.backend_bases.FigureCanvasBase` subclass
        The parent figure canvs for the widget.
    imgplt : ImageWindow
        Window containing image used in measurement.
    image : array
        Image used in measurement/manipulation.
    active : bool
        If False, the widget does not respond to events.
    """

    def __init__(self, image_window, useblit=None, figsize=None):
        self.imgplt = image_window
        self.image = self.imgplt._img

        figsize = plt.rcParam['figure.figsize'] if figsize is None else figsize
        figure = plt.figure(figsize=figsize)
        self.figure = figure
        self.canvas = figure.canvas
        self.ax = figure.add_subplot(111)

        if useblit is None:
            useblit = True if mpl.backends.backend.endswith('Agg') else False
        self.useblit = useblit

        self.active = True
        self.cids = []
        self.artists = []

        self.connect_event('draw_event', self.on_draw)
        self.canvas.mpl_connect('close_event', self.on_close)

    def redraw(self):
        self.canvas.draw_idle()

    def on_draw(self, event):
        """Save image background when blitting.

        The saved image is used to "clear" the figure before redrawing artists.
        """
        if self.useblit:
            bbox = self.imgplt.ax.bbox
            self.img_background = self.imgplt.canvas.copy_from_bbox(bbox)

    def on_close(self, event):
        """Disconnect all artists and events from ImageWindow.

        Note that events must be connected using `self.connect_event` and
        artists must be appended to `self.artists`.
        """
        self.disconnect_image_events()
        self.remove_artists()
        self.imgplt.redraw()

    def ignore(self, event):
        """Return True if event should be ignored.

        This method (or a version of it) should be called at the beginning
        of any event callback.
        """
        return not self.active

    def connect_event(self, event, callback):
        """Connect callback with an event.

        This should be used in lieu of `figure.canvas.mpl_connect` since this
        function stores call back ids for later clean up.
        """
        cid = self.imgplt.connect_event(event, callback)
        self.cids.append(cid)

    def disconnect_image_events(self):
        """Disconnect all events created by this widget."""
        for c in self.cids:
            self.imgplt.disconnect_event(c)

    def remove_artists(self):
        """Disconnect artists that are connected to the *image plot*."""
        for a in self.artists:
            self.imgplt.remove_artist(a)


class LineProfile(Widget):
    """Widget to compute interpolated intensity under a scan line on an image.

    Parameters
    ----------
    image_window : ImageWindow instance.
        Window containing image used in measurement.
    useblit : bool
        If True, use blitting to speed up animation. Only available on some
        backends. If None, set to True when using Agg backend, otherwise False.
    linewidth : float
        Line width for interpolation. Wider lines average over more pixels.
    epsilon : float
        Maximum pixel distance allowed when selecting end point of scan line.
    limits : tuple or {None, 'image', 'dtype'}
        (minimum, maximum) intensity limits for plotted profile. The following
        special values are defined:

            None : rescale based on min/max intensity along selected scan line.
            'image' : fixed scale based on min/max intensity in image.
            'dtype' : fixed scale based on min/max intensity of image dtype.
    """


    def __init__(self, image_window, useblit=None,
                 linewidth=1, epsilon=5, limits='image'):

        Widget.__init__(self, image_window, useblit=useblit, figsize=(8, 3))

        self.linewidth = linewidth
        self.epsilon = epsilon

        if limits == 'image':
            self.limits = (np.min(self.image), np.max(self.image))
        elif limits == 'dtype':
            self.limits = dtype_range[self.image.dtype.type]
        elif limits is None or len(limits) == 2:
            self.limits = limits
        else:
            raise ValueError("Unrecognized `limits`: %s" % limits)

        h, w = self.image.shape

        self._init_end_pts = np.array([[w/3, h/2], [2*w/3, h/2]])
        self.end_pts = self._init_end_pts.copy()

        x, y = np.transpose(self.end_pts)
        self.scan_line = self.imgplt.ax.plot(x, y, 'y-s', markersize=5,
                                             lw=linewidth, alpha=0.5,
                                             solid_capstyle='butt')[0]
        self.artists.append(self.scan_line)

        scan_data = profile_line(self.image, self.end_pts)
        self.profile = self.ax.plot(scan_data, 'k-')[0]
        self._autoscale_view()

        self._active_pt = None

        self.connect_event('key_press_event', self.on_key_press)
        self.connect_event('button_press_event', self.on_mouse_press)
        self.connect_event('button_release_event', self.on_mouse_release)
        self.connect_event('motion_notify_event', self.on_move)
        self.connect_event('scroll_event', self.on_scroll)

        self.imgplt.redraw()
        print self.help()

    def help(self):
        helpstr = ("Line profile tool",
                   "+ and - keys or mouse scroll changes width of scan line.",
                   "Select and drag ends of the scan line to adjust it.")
        return '\n'.join(helpstr)

    def get_profile(self):
        """Return intensity profile of the selected line.

        Returns
        -------
        end_pts: (2, 2) array
            The positions ((x1, y1), (x2, y2)) of the line ends.
        profile: 1d array
            Profile of intensity values.
        """
        end_pts = self.scan_line.get_xydata()
        profile = self.profile.get_ydata()
        return end_pts, profile

    def on_scroll(self, event):
        if not event.inaxes: return
        if event.button == 'up':
            self._thicken_scan_line()
        elif event.button == 'down':
            self._shrink_scan_line()

    def on_key_press(self, event):
        if not event.inaxes: return
        elif event.key == '+':
            self._thicken_scan_line()
        elif event.key == '-':
            self._shrink_scan_line()
        elif event.key == 'r':
            self.reset()

    def _thicken_scan_line(self):
        self.linewidth += 1
        self.line_changed(None, None)

    def _shrink_scan_line(self):
        if self.linewidth > 1:
            self.linewidth -= 1
            self.line_changed(None, None)

    def _autoscale_view(self):
        if self.limits is None:
            self.ax.autoscale_view(tight=True)
        else:
            self.ax.autoscale_view(scaley=False, tight=True)

    def get_pt_under_cursor(self, event):
        """Return index of the end point under cursor, if sufficiently close"""
        xy = np.asarray(self.scan_line.get_xydata())
        xyt = self.scan_line.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        if d[ind] >= self.epsilon:
            ind = None
        return ind

    def on_mouse_press(self, event):
        if event.button != 1: return
        if event.inaxes==None: return
        self._active_pt = self.get_pt_under_cursor(event)

    def on_mouse_release(self, event):
        if event.button != 1: return
        self._active_pt = None

    def on_move(self, event):
        if event.button != 1: return
        if self._active_pt is None: return
        if not self.imgplt.ax.in_axes(event): return
        x,y = event.xdata, event.ydata
        self.line_changed(x, y)

    def reset(self):
        self.end_pts = self._init_end_pts.copy()
        self.scan_line.set_data(np.transpose(self.end_pts))
        self.line_changed(None, None)

    def line_changed(self, x, y):
        if x is not None:
            self.end_pts[self._active_pt, :] = x, y
        self.scan_line.set_data(np.transpose(self.end_pts))
        self.scan_line.set_linewidth(self.linewidth)

        scan = profile_line(self.image, self.end_pts, linewidth=self.linewidth)
        self.profile.set_xdata(np.arange(scan.shape[0]))
        self.profile.set_ydata(scan)

        self.ax.relim()

        if self.useblit:
            self.imgplt.canvas.restore_region(self.img_background)
            self.ax.draw_artist(self.scan_line)
            self.ax.draw_artist(self.profile)
            self.imgplt.canvas.blit(self.imgplt.ax.bbox)

        self._autoscale_view()

        self.imgplt.redraw()
        self.redraw()


def profile_line(img, end_pts, linewidth=1):
    """Return the intensity profile of an image measured along a scan line.

    Parameters
    ----------
    img : 2d array
        The image.
    end_pts: (2, 2) list
        End points ((x1, y1), (x2, y2)) of scan line.
    linewidth: int
        Width of the scan, perpendicular to the line

    Returns
    -------
    return_value : array
        The intensity profile along the scan line. The length of the profile
        is the ceil of the computed length of the scan line.
    """
    point1, point2 = end_pts
    x1, y1 = point1 = np.asarray(point1, dtype = float)
    x2, y2 = point2 = np.asarray(point2, dtype = float)
    dx, dy = point2 - point1

    # Quick calculation if perfectly horizontal or vertical (remove?)
    if x1 == x2:
        pixels = img[min(y1, y2) : max(y1, y2)+1,
                     x1 - linewidth / 2 :  x1 + linewidth / 2 + 1]
        intensities  = pixels.mean(axis = 1)
        return intensities
    elif y1 == y2:
        pixels = img[y1 - linewidth / 2 :  y1 + linewidth / 2 + 1,
                     min(x1, x2) : max(x1, x2)+1]
        intensities = pixels.mean(axis = 0)
        return intensities

    theta = np.arctan2(dy,dx)
    a = dy/dx
    b = y1 - a * x1
    length = np.hypot(dx, dy)

    line_x = np.linspace(min(x1, x2), max(x1, x2), np.ceil(length))
    line_y = line_x * a + b
    y_width = abs(linewidth * np.sin(theta)/2)
    perp_ys = np.array([np.linspace(yi - y_width,
                                    yi + y_width, linewidth) for yi in line_y])
    perp_xs = - a * perp_ys + (line_x +  a * line_y)[:, np.newaxis]

    perp_lines = np.array([perp_ys, perp_xs])
    pixels = ndi.map_coordinates(img, perp_lines)
    intensities = pixels.mean(axis=1)

    return intensities


def mplviewer(image, **kwargs):
    """Return ImageWindow for input image.

    Keyword arguments are passed on to Matplotlib's `imshow` function
    """
    image_window = ImageWindow(image, **kwargs)
    return image_window


if __name__ == '__main__':
    import skimage.io as sio
    from skimage import data

    #image_window = mplviewer(data.camera())
    image_window = sio.imshow(data.camera(), fancy=True)

    # Note: Widget must be assigned to a variable so it isn't garbage collected
    # Maybe LineProfile should save a reference of itself in ImageWindow
    # and then clear itself when closed.
    lp = LineProfile(image_window, limits='dtype')
    plt.show()

