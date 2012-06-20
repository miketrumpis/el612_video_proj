from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches

## def select_rect_object(ax):

##     def onselect(on, off):
##         return [on.xdata, off.xdata, on.ydata, off.ydata]
##     def toggle_selector(event):
##         if event.key in ['Q', 'q'] and toggle_selector.RS.active:
##             print ' RectangleSelector deactivated.'
##             toggle_selector.RS.set_active(False)
##         if event.key in ['A', 'a'] and not toggle_selector.RS.active:
##             print ' RectangleSelector activated.'
##             toggle_selector.RS.set_active(True)
    
##     toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box')
##     connect('key_press_event', toggle_selector)

def draw_rect(plt_image, loc, width, height):
    Ny, Nx = plt_image.get_size()
    lx, ly = loc
    lx -= width/2.0
    ly -= height/2.0
    lyf = max(0, min(ly, Ny))
    height -= abs(lyf - ly)
    lxf = max(0, min(lx, Nx))
    width -= abs(lxf - lx)

    rpoly = patches.Rectangle(
        (lxf, lyf), width, height, fill=False, ec=(1,1,1), lw=3
        )
    plt_image.get_axes().add_artist(rpoly)
    return
