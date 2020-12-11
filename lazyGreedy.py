import heapq

def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy_heap(F, V, B):
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)
    # [_heappush_max(order, (F.inc(sset, index), index)) for index in V]
    cnt = 0
    for index in V:
      _heappush_max(order, (F.inc(sset, index), index))
      cnt += 1

    n_iter = 0
    while order and len(sset) < B:
        n_iter += 1
        if F.curVal == len(F.D):
          # all points covered
          break

        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        # check for uniques elements
        if improv > 0: 
            if not order:
                curVal = F.add(sset, el[1], improv) # NOTE: added "improv"
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1], improv) # NOTE: added "improv"
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)

    return sset, vals
