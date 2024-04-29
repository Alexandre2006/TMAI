class Buffer:
    def __init__(self):
        # Memory
        self.memory = []
        self.maximum_size = 1000000 # TODO: Check if this is too small/large

        # Statistics
        self.test_return_stat = 0.0
        self.train_return_stat = 0.0
        self.test_steps_stat = 0
        self.train_steps_stat = 0
    
    def __len__(self):
        return len(self.memory)

    def __iadd__(self, buffer):
        self.memory += buffer.memory
        self.shorten()

        self.test_return_stat += buffer.test_return_stat
        self.train_return_stat += buffer.train_return_stat
        self.test_steps_stat += buffer.test_steps_stat
        self.train_steps_stat += buffer.train_steps_stat

        return self
    
    def shorten(self):
        length = len(self.memory)
        if length > self.maximum_size:
            print(f"Shortening buffer memory from {length} to {self.maximum_size}")
            self.memory = self.memory[length - self.maximum_size:]
    
    def append(self, sample):
        self.memory.append(sample)
        self.shorten()
    
    def clear_memory(self):
        self.memory = []

shared_buffer = Buffer()
shared_weights = None