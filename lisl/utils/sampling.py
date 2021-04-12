from itertools import cycle, islice

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    # see https://docs.python.org/3/library/itertools.html#recipes
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

def roundrobin_break_early(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E"
    # Recipe credited to George Sakkis
    # see https://docs.python.org/3/library/itertools.html#recipes
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            break
