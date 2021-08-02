import math
import numpy as np
from datetime import timedelta
from dtaidistance import dtw

def diff_smooth(ts):
    dif = ts.diff().dropna()
    td = dif.describe()

    if td["max"]-td['75%'] > 50 or td["min"]-td['25%'] > 50:
        return ts

    high = td['75%'] + 1.5 * (td['75%'] - td['25%'])
    low = td['25%'] - 1.5 * (td['75%'] - td['25%'])

    forbid_index = dif[(dif > high) | (dif < low)].index 
    i = 0
    while i < len(forbid_index) - 1:
        n = 1
        start = forbid_index[i]
        while forbid_index[i+n] == start + timedelta(minutes=n):
            if (i+n) < len(forbid_index)-1:
                n += 1
            else:
                break
        i += n - 1

        end = forbid_index[i]
        value = np.linspace(ts[start - timedelta(minutes=1)], ts[end + timedelta(minutes=1)], n)
        ts[start: end] = value
        i += 1
    
    return ts

def period_check(ts, period, dtw_value = 3):
    cycle = math.floor(len(ts) / period)
    period_split = np.array_split(ts,cycle)

    distance = dtw.distance(period_split[0], period_split[1])
    
    print("distance:", distance)
    if dtw_value > distance:
        return True
    
    return False