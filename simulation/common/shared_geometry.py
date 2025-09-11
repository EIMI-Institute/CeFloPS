# shared_geometry.py
import numpy as np
from multiprocessing import shared_memory

class SharedGeometryConfig:
    disk_load_mode = False

    @classmethod
    def set_mode(cls, mode):
        cls.disk_load_mode = mode
class SharedGeometry:
    def __init__(self, pointarray):
        points = np.asarray(pointarray, dtype=float)
        if SharedGeometryConfig.disk_load_mode:
            # Create shared memory
            self.shm = shared_memory.SharedMemory(create=True, size=points.nbytes)
            self.name = self.shm.name
            self.array = np.ndarray(points.shape, dtype=float, buffer=self.shm.buf)
            self.array[:] = points[:]
            self.shape = points.shape
            self.dtype = points.dtype
        else:
            raise RuntimeError("SharedGeometry must be initialized from disk load")

    def __getstate__(self):
        return {
            "name": self.name,
            "shape": self.shape if hasattr(self,"shape") else None,
            "dtype": self.dtype  if hasattr(self,"dtype") else None
        }

    def __setstate__(self, state): 
        if SharedGeometryConfig.disk_load_mode:#in primary process
            #self.__dict__.update(state)
            self.shm = None
            self.array = None
            self.name=None#dont load the name into the parent process, the parent process should recreate the sharedgeometry!
        else:
            if "dtype" not in state or "shape" not in state:
                assert False
            self.__dict__.update(state)
            if state["name"]!=None:
                self.shm = shared_memory.SharedMemory(name=state["name"]) # bind to sharedmemory
                self.array = np.ndarray(state["shape"], dtype=state["dtype"], buffer=self.shm.buf)
            else:
                print("DONT LOAD NONEVALUE")

            self.name = state["name"]


    def get_points(self):
        """Access the shared array. Attaches if not already connected."""
        return self.array
    def close(self):
        if hasattr(self, "shm"):
            self.shm.close()

    def unlink(self):
        if hasattr(self, "shm"):
            self.shm.unlink()
