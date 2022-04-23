from typing import Union
import numpy as np
from matplotlib.ticker import Locator, Formatter
from matplotlib.axis import Axis
from matplotlib.axes import Axes
from matplotlib.transforms import Transform

"""
The elements in this file allow axes locators and tickers to repersent
some transformed data without doing the transform inverse on the axes
data itself.
"""

class MonkeyPatchingError(NotImplementedError):
    pass

class ProxyAxes:
    def __init__(self, proxy_axis: "TransformedProxyAxis"):
        self._proxy_axis = proxy_axis
    
    @property
    def name(self):
        return self._proxy_axis._axis.axes.name

    def get_xaxis_transform(self):
        if self._proxy_axis.__name__ != "xaxis":
            raise MonkeyPatchingError(
                "Requested x axis transform for a y axis ticker")
        return self._proxy_axis.get_transform()

    def get_yaxis_transform(self):
        if self._proxy_axis.__name__ != "yaxis":
            raise MonkeyPatchingError(
                "Requested y axis transform for a x axis ticker")
        return self._proxy_axis.get_transform()
    
def map_ys(f_xy, ys):
    xys = np.transpose([np.zeros(np.shape(ys)), ys])
    return np.asarray(f_xy(xys))[:,1]

def map_xs(f_xy, xs):
    xys = np.transpose([xs, np.zeros(np.shape(xs))])
    return np.asanyarray(f_xy(xys))[:,0]

class TransformedProxyAxis:
    def __init__(self, axis: Axis, transform: Transform):
        self._transform = transform
        self._axis = axis

    def _transform_on_axis(self, data, inverse=False):
        t = self._transform if not inverse else self._transform.inverted()
        mapf = {"yaxis": map_ys, "xaxis": map_xs}.get(self.__name__, None)
        if mapf is not None:
            return mapf(t.transform, data)
        raise MonkeyPatchingError(
            f"Proxied axis __name__ is {self.__name__!r}, expected 'xaxis' or 'yaxis'")

    def _trn(self, data):
        return self._transform_on_axis(data)

    def _inv(self, data):
        return self._transform_on_axis(data, inverse=True)

    @property
    def axes(self):
        return ProxyAxes(self)

    @property
    def __name__(self):
        return self._axis.__name__

    def get_scale(self):
        return self._axis.get_scale()

    def get_majorticklocs(self):
        return self._inv(self._axis.get_majorticklocs())

    def get_transform(self):
        # I'm not sure about the order of these two
        return self._transform.inverted() + self._axis.get_trasnform()

    def get_view_interval(self):
        return self._inv(self._axis.get_view_interval())

    def set_view_interval(self, vmin, vmax):
        return self._axis.set_view_interval(*self._trn((vmin, vmax)))

    def get_data_interval(self):
        return self._inv(self._axis.get_data_interval())

    def set_data_interval(self, vmin, vmax):
        return self._axis.set_data_interval(*self._trn((vmin, vmax)))

    def get_minpos(self):
        minpos, = self._inv([self._axis.get_minpos()])
        return minpos

    def get_tick_space(self):
        return self._axis.get_tick_space()

# Couldn't you have just had a function take the data, view limits and some parameters
# like projection and compute the new limits and ticks?

class TransformedTickLocatorWrapper(Locator):
    def __init__(self, locator: Locator, transform: Transform):
        self._locator = locator
        self._transform = transform

    def set_axis(self, axis):
        self._axis = axis
        self._locator.set_axis(self._proxy_axis)
        
    @property
    def _proxy_axis(self):
        return TransformedProxyAxis(self._axis, self._transform)

    @property
    def axis(self):
        return self._axis

    def tick_values(self, vmin, vmax):
        return self._proxy_axis._trn(
            self._locator.tick_values(
                *self._proxy_axis._inv([vmin, vmax]))
        )
        
    def view_limits(self, dmin, dmax):
        return self._proxy_axis._trn(
            self._locator.view_limits(
                *self._proxy_axis._inv([dmin, dmax]))
        )
    
    def __call__(self):
        return self._proxy_axis._trn(
            self._locator.__call__()
        )

    def raise_if_exceeds(self, locs):
        return self._proxy_axis._trn(
            self._locator.raise_if_exceeds(
                *self._proxy_axis._inv(locs))
        )

    def nonsingular(self, v0, v1):
        return self._proxy_axis._trn(
            self._locator.nonsingular(
                *self._proxy_axis._inv([v0, v1]))
        )

# It would have been to easy for these people to just have a functon take a list
# of values, some parameters and return the string values for the ticks. It's really
# amazing how much unnecessary complexity OOP adds

class TransformedFormatterWrapper(Formatter):
    def __init__(self, formatter: Formatter, transform: Transform):
        self._formatter = formatter
        self._transform = transform

    @property
    def axis(self):
        return self._axis
    
    def set_axis(self, axis):
        self._axis = axis
        self._formatter.set_axis(self._proxy_axis)
        
    @property
    def _proxy_axis(self):
        return TransformedProxyAxis(self._axis, self._transform)

    @property
    def locs(self):
        return self._proxy_axis._trn(self._formatter.locs)

    def __call__(self, x, pos=None):
        val, = self._proxy_axis._inv([x])
        return self._formatter.__call__(val, pos)

    def format_ticks(self, values):
        return self._formatter.format_ticks(self._proxy_axis._inv(values))

    def format_data(self, value):
        val, = self._proxy_axis._inv([value])
        return self._formatter.format_data(val)

    def format_data_short(self, value):
        val, = self._proxy_axis._inv([value])
        return self._formatter.format_data_short(val)

    def get_offset(self):
        return self._formatter.get_offset()

    def set_locs(self, locs):
        self._formatter.set_locs(self._proxy_axis._inv(locs))

def set_transformed_tickers(ax: Union[Axis, Axes], transform: Transform):
    if isinstance(ax, Axes):
        set_transformed_tickers(ax.xaxis, transform)
        set_transformed_tickers(ax.yaxis, transform)
        return
    ax.set_major_locator(
        TransformedTickLocatorWrapper(
            ax.get_major_locator(),
            transform
        ))
    ax.set_minor_locator(
        TransformedTickLocatorWrapper(
            ax.get_minor_locator(),
            transform
        ))
    ax.set_major_formatter(
        TransformedFormatterWrapper(
            ax.get_major_formatter(),
            transform
        ))
    ax.set_minor_formatter(
        TransformedFormatterWrapper(
            ax.get_minor_formatter(),
            transform
        ))
