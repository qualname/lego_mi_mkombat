import operator


class SegmentTree:
    def __init__(self, capacity, operation, neutral_elem):
        self.capacity = capacity
        self.operation = operation
        self.values = [neutral_elem] * (2 * self.capacity)

    def query(self, start, end):
        return self._q_recursive(start, end - 1, 1, 0, self.capacity - 1)

    def _q_recursive(self, start, end, node, start_node, end_node):
        if start == start_node and end == end_node:
            return self.values[node]

        middle = (start_node + end_node) // 2

        if end <= middle:
            return self._q_recursive(start, end, 2 * node, start_node, middle)
        elif middle + 1 <= start:
            return self._q_recursive(start, end, 2 * node + 1, middle + 1, end_node)
        else:
            return self.operation(
                self._q_recursive(start, middle, 2 * node, start_node, middle),
                self._q_recursive(middle + 1, end, 2 * node + 1, middle + 1, end_node),
            )

    def _q_iterative(self, left, right):
        left += self.capacity
        right += self.capacity

        acc = self.values[0]
        while left < right:
            if left % 2 == 1:
                acc = self.operation(acc, self.values[left])
                left += 1
            if right % 2 == 1:
                right -= 1
                acc = self.operation(acc, self.values[right])

            left //= 2
            right //= 2

        return acc

    def __getitem__(self, idx):
        return self.values[self.capacity + idx]

    def __setitem__(self, idx, val):
        idx += self.capacity
        self.values[idx] = val

        while idx > 1:
            idx //= 2
            self.values[idx] = self.operation(
                self.values[2 * idx], self.values[2 * idx + 1]
            )


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=min, neutral_elem=float('inf'))

    def __call__(self, start=0, end=self.capacity):
        return super().query(start, end)


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=operator.add, neutral_elem=0.0)

    def __call__(self, start=0, end=self.capacity):
        return super().query(start, end)
