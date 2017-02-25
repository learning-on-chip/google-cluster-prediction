import numpy as np
import queue
import socket
import support
import threading


class Manager:
    def __init__(self, config):
        self.test_schedule = Schedule(config.test_schedule)
        self.backup_schedule = Schedule(config.backup_schedule)
        self.show_schedule = Schedule(config.show_schedule)
        self.show_address = config.show_address
        self.listeners = {}
        self.lock = threading.Lock()
        worker = threading.Thread(target=self._show_server, daemon=True)
        worker.start()

    def on_show(self, sample, y_hat, offset):
        count0 = sample.shape[0]
        count1 = count0 - offset
        count2 = y_hat.shape[0]
        count0 = count0 + count2
        message = (np.array([count0, count1, count2]),
                   sample[offset:, :], y_hat)
        with self.lock:
            for listener in self.listeners:
                listener.put(message)
            return len(self.listeners) > 0

    def should_backup(self, time):
        return self.backup_schedule.should(time)

    def should_show(self, time):
        return len(self.listeners) > 0 and self.show_schedule.should(time)

    def should_test(self, time):
        return self.test_schedule.should(time)

    def should_train(self, _):
        return True

    def _show_client(self, connection, address):
        support.log(self, 'New listener: {}', address)
        listener = queue.Queue()
        with self.lock:
            self.listeners[listener] = True
        try:
            client = connection.makefile(mode='w')
            while True:
                values = []
                for chunk in listener.get():
                    values.extend([str(value) for value in chunk.flatten()])
                client.write(','.join(values) + '\n')
                client.flush()
        except Exception as e:
            support.log(self, 'Disconnected listener: {} ({})', address, e)
        with self.lock:
            del self.listeners[listener]

    def _show_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.show_address)
        server.listen(1)
        support.log(self, 'Show address: {}', self.show_address)
        while True:
            try:
                connection, address = server.accept()
                worker = threading.Thread(target=self._show_client,
                                          args=(connection, address),
                                          daemon=True)
                worker.start()
            except Exception as e:
                support.log(self, 'Exception: {}', e)


class Schedule:
    def __init__(self, schedule):
        self.schedule = np.cumsum(schedule)

    def should(self, time):
        time = time % self.schedule[-1] + 1
        phase = np.nonzero(self.schedule >= time)[0][0]
        return phase % 2 == 1
