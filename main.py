import sys
from math import *
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

sys.setrecursionlimit(100001)
INF = 10000000000000
const_color = {
    'margrr': 0,
    'margrg': 0,
    'margrb': -10,
    'margbr': 10,
    'margbg': 0,
    'margbb': 5,
    'radr': 24000,
    'radb': 39000
}
const_dfs = {
    'fact': 1
}

def dfc(ind, data):
    r, g, b = data['r'][ind], data['g'][ind], data['b'][ind]
    radred = (255 - r + const_color['margrr'])**2 + (g + const_color['margrg'])**2 + (b + const_color['margrb'])**2
    radblu = (255 - b + const_color['margbb'])**2 + (g + const_color['margbg'])**2 + (r + const_color['margbr'])**2
    if radblu <= const_color['radb']:
        return 2
    elif radred <= const_color['radr']:
        return 1
    else:
        return 0

def loaddata(file, data):
    data['x'] = []
    data['y'] = []
    data['z'] = []
    data['r'] = []
    data['g'] = []
    data['b'] = []
    data['c'] = []
    data['val'] = set()
    data['col'] = {}
    data['used'] = set()

    with open(file, 'r') as f:
        lines = f.readlines()

    data['n'] = len(lines)
    for line in lines:
        a = list(map(int, line.split()))
        x, y, z, r, g, b = a[:6]
        data['x'].append(x)
        data['y'].append(y)
        data['z'].append(z)
        data['r'].append(r)
        data['g'].append(g)
        data['b'].append(b)
        col = dfc(len(data['x'])-1, data)
        data['c'].append(col)
        data['val'].add((x, y, z))
        data['col'][(x, y, z)] = col

def draw(ax, data, freq=4, draw_grey=True):
    x_red, y_red = [], []
    x_blu, y_blu = [], []
    x_gry, y_gry = [], []

    for i in range(0, data['n'], freq):
        col = data['c'][i]
        x = -data['x'][i]
        y = -data['y'][i]
        if col == 0:
            x_gry.append(x)
            y_gry.append(y)
        elif col == 1:
            x_red.append(x)
            y_red.append(y)
        elif col == 2:
            x_blu.append(x)
            y_blu.append(y)

    if x_red:
        ax.scatter(x_red, y_red, c='red', s=3, alpha=1.0, label='Red')
    if x_blu:
        ax.scatter(x_blu, y_blu, c='blue', s=3, alpha=1.0, label='Blue')
    if draw_grey and x_gry:
        ax.scatter(x_gry, y_gry, c='gray', s=2, alpha=0.8, label='Grey')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

def getrect(start_ind, curcol, data):
    q = deque()
    start = (data['x'][start_ind], data['y'][start_ind], data['z'][start_ind])
    q.append(start)
    data['used'].add(start)
    rect = []
    while q:
        xx, yy, zz = q.popleft()
        rect.append((xx, yy, zz))
        for i in range(-const_dfs['fact'], const_dfs['fact']+1):
            for j in range(-const_dfs['fact'], const_dfs['fact']+1):
                for k in range(-const_dfs['fact'], const_dfs['fact']+1):
                    if i == j == k == 0:
                        continue
                    nb = (xx+i, yy+j, zz+k)
                    if nb in data['val'] and nb not in data['used'] and data['col'][nb] == curcol:
                        q.append(nb)
                        data['used'].add(nb)
    return rect

def bbrect(rect, ang):
    prop = getprop_rect(rect)
    tg1 = tan(ang * pi / 180)
    tg2 = tan(ang * pi / 180 - pi / 2)
    b1min = INF
    b1max = -INF
    b2min = INF
    b2max = -INF
    for (x, y, z) in rect:
        b1 = tg1 * x - y
        b2 = tg2 * x - y
        b1min = min(b1min, b1)
        b1max = max(b1max, b1)
        b2min = min(b2min, b2)
        b2max = max(b2max, b2)
    x1 = (b2max - b1max) / (tg1 - tg2)
    x2 = (b2max - b1min) / (tg1 - tg2)
    x3 = (b2min - b1max) / (tg1 - tg2)
    x4 = (b2min - b1min) / (tg1 - tg2)
    y1 = tg1 * x1 + b1max
    y2 = tg1 * x2 + b1min
    y3 = tg1 * x3 + b1max
    y4 = tg1 * x4 + b1min
    dist1 = hypot(x1-x2, y1-y2)
    dist2 = hypot(x1-x3, y1-y3)
    q = 1 if dist2 >= dist1 else -1
    return [dist1*dist2, q, ang, (x1,y1), (x2,y2), (x3,y3), (x4,y4)]

def getprop_rect(rect):
    n = len(rect)
    xsum = sum(p[0] for p in rect) / n
    ysum = sum(p[1] for p in rect) / n
    zsum = sum(p[2] for p in rect) / n
    return (xsum, ysum, zsum)

def findprop_rect(rect):
    mini = [INF, None, None]
    for ang in range(1, 180):
        if ang == 90:
            continue
        k = bbrect(rect, ang)
        if k[0] < mini[0]:
            mini = k
            if mini[1] == -1:
                mini[2] = ang-90 if ang>90 else ang+90
            else:
                mini[2] = ang
    mini[2] = 90 - mini[2]
    return mini

def main():
    FILES = [f'data/{i}.{j}' for i in range(2, 6) for j in range(1, 4)]

    n_files = len(FILES)
    if n_files == 0:
        print("No files to process.")
        return

    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    axes_flat = axes.flatten()

    for i in range(n_files, len(axes_flat)):
        axes_flat[i].axis('off')

    for idx, fname in enumerate(FILES[:n_rows*n_cols]):
        ax = axes_flat[idx]
        ax.set_title(fname)

        data = {
            'x': [], 'y': [], 'z': [],
            'r': [], 'g': [], 'b': [], 'c': [],
            'val': set(), 'col': {}, 'used': set(), 'n': 0
        }

        try:
            loaddata(fname, data)
        except FileNotFoundError:
            ax.text(0.5, 0.5, 'File not found', ha='center', va='center')
            continue

        draw(ax, data, freq=4, draw_grey=True)

        rects = []
        for i in range(data['n']):
            col = data['c'][i]
            if col in (1,2):
                pos = (data['x'][i], data['y'][i], data['z'][i])
                if pos not in data['used']:
                    rect = getrect(i, col, data)
                    rects.append(rect)

        rects.sort(key=len, reverse=True)
        rects = [r for r in rects if len(r) >= 250]

        rect_corners = []
        rect_center = []
        rect_radius = []
        for rect in rects:
            prop = findprop_rect(rect)
            corners = [prop[3], prop[4], prop[5], prop[6]]
            x2, y2 = prop[4]
            x3, y3 = prop[5]
            center = ((x2+x3)/2, (y2+y3)/2)
            radius = hypot(x2-center[0], y2-center[1])
            rect_corners.append(corners)
            rect_center.append(center)
            rect_radius.append(radius)
            xs = [corners[0][0], corners[1][0], corners[3][0], corners[2][0], corners[0][0]]
            ys = [corners[0][1], corners[1][1], corners[3][1], corners[2][1], corners[0][1]]
            ax.plot(xs, ys, 'k-', linewidth=1)
            ax.plot(center[0], center[1], 'ro', markersize=3)

        n_comp = len(rect_corners)
        for i in range(n_comp):
            flag = True
            for j in range(n_comp):
                if i == j:
                    continue
                for k in range(4):
                    p1 = rect_corners[j][k]
                    p2 = rect_corners[j][(k+1)%4]
                    dx = p2[0] - p1[0]
                    if abs(dx) < 1e-12:
                        continue
                    q = (p2[1] - p1[1]) / dx
                    b = p1[1] - q * p1[0]

                    x0, y0 = rect_center[i]
                    r = rect_radius[i]
                    a = q*q + 1
                    bb = 2*b*q - 2*y0*q - 2*x0
                    c = x0*x0 - 2*y0*b + b*b + y0*y0 - r*r
                    D = bb*bb - 4*a*c
                    if D < 0:
                        continue
                    sqrtD = sqrt(D)
                    x1 = (-bb + sqrtD) / (2*a)
                    x2 = (-bb - sqrtD) / (2*a)
                    minx = min(p1[0], p2[0])
                    maxx = max(p1[0], p2[0])
                    if minx <= x1 <= maxx:
                        # ax.plot(x1, x1*q + b, 'ro', markersize=3)
                        flag = False
                    if minx <= x2 <= maxx and abs(x1-x2) > 1e-6:
                        # ax.plot(x2, x2*q + b, 'ro', markersize=3)
                        flag = False
                if not flag:
                    break
        if idx == 0:
            ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()