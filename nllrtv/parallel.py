import inspect
from functools import partial

from dask.diagnostics import ProgressBar
import dask.array

from . import aggregate


class daskify(object):
    """ parallizes a stack filtering function using DASK"""

    def __init__(self, chunks, new_axis, aggr=False):
        """
        Parameters
        ----------

        chunks: tuple of ints
            spatial chunk dimensions
        """
        self.chunks = chunks
        self.new_axis = new_axis
        self.aggr = aggr

    def __call__(self, func):
        self.func = func

        def wrapped(stack, *args, **kwargs):
            func = self.func
            # transform args to keyword arguments for partial application
            args = dict(zip(inspect.signature(func).parameters.keys(), [stack, *args]))
            args.pop("stack")  # remove stack
            kwargs.update(args)
            func_pa = partial(func, **kwargs)

            stack_da = dask.array.from_array(
                stack, chunks=(stack.shape[0], *self.chunks)
            )
            depth_map = (0, *tuple(x // 2 for x in kwargs["win_shape"][1:]))
            if not self.new_axis:
                depth_trim = depth_map
            else:
                depth_trim = (*((0,) * len(self.new_axis)), *depth_map)
            depth_trim = dict(zip(range(len(depth_trim)), depth_trim))

            with dask.config.set(scheduler="processes"):
                with ProgressBar():
                    temp_arr = dask.array.map_overlap(
                        stack_da,
                        func_pa,
                        depth_map,
                        new_axis=self.new_axis,
                        trim=False,
                        dtype=stack_da.dtype,
                    )

                    if self.aggr:
                        # Aggregate overlapping parts.
                        # Works only with chunks of equal size
                        chunk_shape = tuple((x[0] for x in temp_arr.chunks))
                        temp_arr = temp_arr.compute()
                        temp_arr_aggr, aggr_cnt = aggregate.overlap_aggregate(
                            temp_arr, chunk_shape, depth_trim
                        )

                        # convert to dask array so that overlapping parts can be trimmed with existing code
                        temp_arr = dask.array.from_array(
                            temp_arr_aggr / aggr_cnt, chunks=chunk_shape
                        )

                    return dask.array.overlap.trim_internal(
                        temp_arr, depth_trim
                    ).compute()

        wrapped.__signature__ = inspect.signature(func)
        wrapped.__name__ = func.__name__
        wrapped.__doc__ = func.__doc__

        return wrapped
