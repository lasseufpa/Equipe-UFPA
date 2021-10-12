import threading

class Collector(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._data = None
    
    def callback(self, data):
        with self._lock:
            self._data = self._process(data)
    
    def _process(self, data):
        return data
    
    @property
    def data(self):
        with self._lock:
            return self._data
    
    @data.setter
    def data(self, value):
        self._data = value