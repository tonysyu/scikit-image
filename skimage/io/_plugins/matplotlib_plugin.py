import matplotlib.pyplot as plt

def imshow(image, fancy=False, **kwargs):
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', 'gray')

    if fancy:
        from .mplvi import mplviewer
        return mplviewer(image, **kwargs)
    else:
        ax = plt.axes()
        ax.imshow(image, **kwargs)


imread = plt.imread
show = plt.show

def _app_show():
    show()
