from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# === Constantes y matemática de mecanismo focal (adaptado de obspy.imaging.beachball) ===
# Referencia: obspy/imaging/beachball.py (Robert Barsch et al.), a su vez adaptado de
# bb.m (Michael, Ji, Boyd) y ps_meca/utilmeca.c (Generic Mapping Tools).
_D2R = np.pi / 180.0
_EPSILON = 1e-5  # mismo valor que obspy.imaging.beachball.EPSILON

_PrincipalAxis = namedtuple("_PrincipalAxis", ["val", "azimuth", "plunge"])
_NodalPlane = namedtuple("_NodalPlane", ["strike", "dip", "rake"])


def _ned_to_use(M):
    """Convierte un tensor de momento de North-East-Down (Aki & Richards) a
    Up-South-East (Harvard/GCMT), la convención que usan las fórmulas de obspy.
    Mrr=Mzz, Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy.
    """
    Mxx, Mxy, Mxz = M[0, 0], M[0, 1], M[0, 2]
    Myy, Myz = M[1, 1], M[1, 2]
    Mzz = M[2, 2]
    Mrr, Mtt, Mpp = Mzz, Mxx, Myy
    Mrt, Mrp, Mtp = Mxz, -Myz, -Mxy
    return np.array([[Mrr, Mrt, Mrp],
                      [Mrt, Mtt, Mtp],
                      [Mrp, Mtp, Mpp]])


def _mt2axes(mt_use):
    """Ejes principales T, N, P de un tensor de momento (convención USE)."""
    d, v = np.linalg.eigh(mt_use)
    pl = np.arcsin(-v[0])
    az = np.arctan2(v[2], -v[1])
    for i in range(3):
        if pl[i] <= 0:
            pl[i] = -pl[i]
            az[i] += np.pi
        if az[i] < 0:
            az[i] += 2 * np.pi
        if az[i] > 2 * np.pi:
            az[i] -= 2 * np.pi
    pl = np.degrees(pl)
    az = np.degrees(az)
    t = _PrincipalAxis(d[2], az[2], pl[2])
    n = _PrincipalAxis(d[1], az[1], pl[1])
    p = _PrincipalAxis(d[0], az[0], pl[0])
    return t, n, p


def _is_pure_dc(T, N, P, eps=_EPSILON):
    return abs(N.val) < eps and abs(T.val + P.val) < eps


def _strike_dip(n, e, u):
    if u < 0:
        n, e, u = -n, -e, -u
    strike = np.degrees(np.arctan2(e, n)) - 90
    strike %= 360
    x = np.sqrt(n ** 2 + e ** 2)
    dip = np.degrees(np.arctan2(x, u))
    return strike, dip


def _aux_plane(s1, d1, r1):
    z = np.radians(s1 + 90)
    z2 = np.radians(d1)
    z3 = np.radians(r1)
    # vector de deslizamiento del plano 1
    sl1 = -np.cos(z3) * np.cos(z) - np.sin(z3) * np.sin(z) * np.cos(z2)
    sl2 = np.cos(z3) * np.sin(z) - np.sin(z3) * np.cos(z) * np.cos(z2)
    sl3 = np.sin(z3) * np.sin(z2)
    strike, dip = _strike_dip(sl2, sl1, sl3)

    n1 = np.sin(z) * np.sin(z2)  # normal al plano 1
    n2 = np.cos(z) * np.sin(z2)
    h1 = -sl2  # vector de rumbo del plano 2 (h3 = 0 siempre)
    h2 = sl1

    zz = (h1 * n1 + h2 * n2) / np.sqrt(h1 ** 2 + h2 ** 2)
    zz = np.clip(zz, -1.0, 1.0)
    zz = np.arccos(zz)
    rake = np.degrees(zz) if sl3 > 0 else -np.degrees(zz)
    return strike, dip, rake


def _tdl(an, bn):
    """Convierte vector normal a la falla `an` y vector de deslizamiento `bn`
    en (strike, dip, rake). Transcripción de bb.m (Michael, Ji, Boyd) tal
    como aparece en obspy.imaging.beachball.tdl."""
    xn, yn, zn = an[0], an[1], an[2]
    xe, ye, ze = bn[0], bn[1], bn[2]
    aaa = 1.0e-6
    con = 180.0 / np.pi
    if np.fabs(zn) < aaa:
        fd = 90.
        axn = min(np.fabs(xn), 1.0)
        ft = np.arcsin(axn) * con
        st, ct = -xn, yn
        if st >= 0. and ct < 0:
            ft = 180. - ft
        if st < 0. and ct <= 0:
            ft = 180. + ft
        if st < 0. and ct > 0:
            ft = 360. - ft
        fl = np.arcsin(abs(ze)) * con
        sl = -ze
        if np.fabs(xn) < aaa:
            cl = xe / yn
        else:
            cl = -ye / xn
        if sl >= 0. and cl < 0:
            fl = 180. - fl
        if sl < 0. and cl <= 0:
            fl = fl - 180.
        if sl < 0. and cl > 0:
            fl = -fl
    else:
        if -zn > 1.0:
            zn = -1.0
        fdh = np.arccos(-zn)
        fd = fdh * con
        sd = np.sin(fdh)
        if sd == 0:
            return None
        st = -xn / sd
        ct = yn / sd
        sx = min(np.fabs(st), 1.0)
        ft = np.arcsin(sx) * con
        if st >= 0. and ct < 0:
            ft = 180. - ft
        if st < 0. and ct <= 0:
            ft = 180. + ft
        if st < 0. and ct > 0:
            ft = 360. - ft
        sl = -ze / sd
        sx = min(np.fabs(sl), 1.0)
        fl = np.arcsin(sx) * con
        if st == 0:
            cl = xe / ct
        else:
            xxx = yn * zn * ze / sd / sd + ye
            cl = -sd * xxx / xn
            if ct == 0:
                cl = ye / st
        if sl >= 0. and cl < 0:
            fl = 180. - fl
        if sl < 0. and cl <= 0:
            fl = fl - 180.
        if sl < 0. and cl > 0:
            fl = -fl
    return ft, fd, fl


def _mt2plane(mt_use):
    """Un plano nodal del tensor de momento (convención USE)."""
    d, v = np.linalg.eig(mt_use)
    d = np.array([d[1], d[0], d[2]])
    v = np.array([[v[1, 1], -v[1, 0], -v[1, 2]],
                  [v[2, 1], -v[2, 0], -v[2, 2]],
                  [-v[0, 1], v[0, 0], v[0, 2]]])
    imax, imin = d.argmax(), d.argmin()
    ae = (v[:, imax] + v[:, imin]) / np.sqrt(2.0)
    an = (v[:, imax] - v[:, imin]) / np.sqrt(2.0)
    aer = np.sqrt(np.sum(ae ** 2))
    anr = np.sqrt(np.sum(an ** 2))
    ae = ae / aer
    an = an / anr if anr else np.array([np.nan, np.nan, np.nan])
    if an[2] <= 0.:
        an1, ae1 = an, ae
    else:
        an1, ae1 = -an, -ae
    ft, fd, fl = _tdl(an1, ae1)
    return _NodalPlane(360 - ft, fd, 180 - fl)


def _pol2cart(th, r):
    return r * np.cos(th), r * np.sin(th)


def _dc_regions(strike1, dip1, rake1, radius=1.0):
    """Los dos polígonos (compresión/dilatación) de una doble-cupla pura.
    Transcripción de la geometría de obspy.imaging.beachball.plot_dc."""
    s_1, d_1, r_1 = strike1, dip1, rake1

    m = 0
    if r_1 > 180:
        r_1 -= 180
        m = 1
    if r_1 < 0:
        r_1 += 180
        m = 1

    s_2, d_2, _r_2 = _aux_plane(s_1, d_1, r_1)

    if d_1 >= 90:
        d_1 = 89.9999
    if d_2 >= 90:
        d_2 = 89.9999

    phi = np.arange(0, np.pi, .01)
    l1 = np.sqrt((90 - d_1) ** 2 / (
        np.sin(phi) ** 2 + np.cos(phi) ** 2 * (90 - d_1) ** 2 / 90 ** 2))
    l2 = np.sqrt((90 - d_2) ** 2 / (
        np.sin(phi) ** 2 + np.cos(phi) ** 2 * (90 - d_2) ** 2 / 90 ** 2))

    regions = []
    for m_ in ((m + 1) % 2, m):
        inc = 1
        x_1, y_1 = _pol2cart(phi + np.radians(s_1), l1)

        if m_ == 1:
            lo, hi = s_1 - 180, s_2
            if lo > hi:
                inc = -1
            th1 = np.arange(s_1 - 180, s_2, inc)
            xs_1, ys_1 = _pol2cart(np.radians(th1), 90 * np.ones_like(th1))
            x_2, y_2 = _pol2cart(phi + np.radians(s_2), l2)
            th2 = np.arange(s_2 + 180, s_1, -inc)
        else:
            hi, lo = s_1 - 180, s_2 - 180
            if lo > hi:
                inc = -1
            th1 = np.arange(hi, lo, -inc)
            xs_1, ys_1 = _pol2cart(np.radians(th1), 90 * np.ones_like(th1))
            x_2, y_2 = _pol2cart(phi + np.radians(s_2), l2)
            x_2, y_2 = x_2[::-1], y_2[::-1]
            th2 = np.arange(s_2, s_1, inc)
        xs_2, ys_2 = _pol2cart(np.radians(th2), 90 * np.ones_like(th2))

        x_all = np.concatenate((x_1, xs_1, x_2, xs_2)) * radius / 90
        y_all = np.concatenate((y_1, ys_1, y_2, ys_2)) * radius / 90

        # obspy arma el parche como xy2patch(y, x, ...): la X en pantalla es
        # la componente "y" de pol2cart y la Y en pantalla es la "x".
        regions.append(np.column_stack((y_all, x_all)))
    return regions


def _mt_regions(T, N, P, plot_zerotrace=True, radius=1.0):
    """Regiones (disco de fondo + hasta 3 polígonos de línea nodal) para un
    tensor de momento general (no doble-cupla pura). Transcripción de la
    geometría de obspy.imaging.beachball.plot_mt (sin la lógica de color)."""
    b = 1
    big_iso = 0
    j, j2, j3, n = 1, 0, 0, 0
    azi = np.zeros((3, 2))
    x = np.zeros(400); y = np.zeros(400)
    x2 = np.zeros(400); y2 = np.zeros(400)
    x3 = np.zeros(400); y3 = np.zeros(400)
    xp1 = np.zeros(800); yp1 = np.zeros(800)
    xp2 = np.zeros(400); yp2 = np.zeros(400)

    a = np.array([T.azimuth, N.azimuth, P.azimuth], dtype=float)
    p = np.array([T.plunge, N.plunge, P.plunge], dtype=float)
    v = np.array([T.val, N.val, P.val], dtype=float)

    vi = (v[0] + v[1] + v[2]) / 3.
    v = v - vi

    radius_size = radius

    if np.fabs(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) < _EPSILON:
        # implosión/explosión pura: un solo disco
        return [("disk", None)]

    if plot_zerotrace:
        vi = 0.

    isotestv0 = 0
    isotestv2 = 0
    for ii in range(360):
        fir = ii * _D2R
        f = -v[1] / v[0]
        iso = vi / v[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            s2alphan = (2. + 2. * iso) / (3. + (1. - 2. * f) * np.cos(2. * fir))
        if s2alphan > 1.:
            isotestv0 += 1
        f = -v[1] / v[2]
        iso = vi / v[2]
        with np.errstate(divide='ignore', invalid='ignore'):
            s2alphan = (2. + 2. * iso) / (3. + (1. - 2. * f) * np.cos(2. * fir))
        if s2alphan > 1.:
            isotestv2 += 1

    if isotestv0 == 0:
        d, m = 0, 2
    elif isotestv2 == 0:
        d, m = 2, 0

    f = -v[1] / v[d]
    iso = vi / v[d]

    # Cliff Frohlich, Seismological Research Letters, Vol 7, Jan-Feb 1996:
    # fuera de este rango isotrópico no hay nodos (P) en absoluto.
    if iso <= -1 or iso >= 1 - f:
        return [("disk", None)]

    spd, cpd = np.sin(p[d] * _D2R), np.cos(p[d] * _D2R)
    spb, cpb = np.sin(p[b] * _D2R), np.cos(p[b] * _D2R)
    spm, cpm = np.sin(p[m] * _D2R), np.cos(p[m] * _D2R)
    sad, cad = np.sin(a[d] * _D2R), np.cos(a[d] * _D2R)
    sab, cab = np.sin(a[b] * _D2R), np.cos(a[b] * _D2R)
    sam, cam = np.sin(a[m] * _D2R), np.cos(a[m] * _D2R)

    azp = 0.0
    az = 0.0
    for i in range(360):
        fir = i * _D2R
        with np.errstate(divide='ignore', invalid='ignore'):
            s2alphan = (2. + 2. * iso) / (3. + (1. - 2. * f) * np.cos(2. * fir))
        if s2alphan > 1.:
            big_iso += 1
        else:
            alphan = np.arcsin(np.sqrt(s2alphan))
            sfi, cfi = np.sin(fir), np.cos(fir)
            san, can = np.sin(alphan), np.cos(alphan)

            xz = can * spd + san * sfi * spb + san * cfi * spm
            xn = can * cpd * cad + san * sfi * cpb * cab + san * cfi * cpm * cam
            xe = can * cpd * sad + san * sfi * cpb * sab + san * cfi * cpm * sam

            if np.fabs(xn) < _EPSILON and np.fabs(xe) < _EPSILON:
                takeoff, az = 0., 0.
            else:
                az = np.arctan2(xe, xn)
                if az < 0.:
                    az += 2. * np.pi
                takeoff = np.arccos(xz / np.sqrt(xz ** 2 + xn ** 2 + xe ** 2))
            if takeoff > np.pi / 2.:
                takeoff = np.pi - takeoff
                az += np.pi
                if az > 2. * np.pi:
                    az -= 2. * np.pi
            r = np.sqrt(2) * np.sin(takeoff / 2.)
            si, co = np.sin(az), np.cos(az)
            if i == 0:
                azi[0][0] = az
                x[0] = radius_size * r * si
                y[0] = radius_size * r * co
                azp = az
            else:
                if np.fabs(np.fabs(az - azp) - np.pi) < _D2R * 10. and takeoff > 80. * _D2R:
                    azi[n][1] = azp
                    n += 1
                    azi[n][0] = az
                if np.fabs(np.fabs(az - azp) - 2. * np.pi) < _D2R * 2.:
                    if azp < az:
                        azi[n][0] += 2. * np.pi
                    else:
                        azi[n][0] -= 2. * np.pi
                if n == 0:
                    x[j] = radius_size * r * si
                    y[j] = radius_size * r * co
                    j += 1
                elif n == 1:
                    x2[j2] = radius_size * r * si
                    y2[j2] = radius_size * r * co
                    j2 += 1
                elif n == 2:
                    x3[j3] = radius_size * r * si
                    y3[j3] = radius_size * r * co
                    j3 += 1
                azp = az
    azi[n][1] = az

    regions = [("disk", None)]

    if n == 0:
        regions.append(("polygon", np.column_stack((x[0:360], y[0:360]))))
        return regions

    if n == 1:
        if big_iso > 0:
            # nota: obspy copia aquí desde `x`, no `x2`, tal cual el original
            for i in range(j2):
                xp1[i] = x[i]
                yp1[i] = y[i]
            for ii in range(j):
                xp1[i] = x2[ii]
                i += 1
                yp1[i] = y2[ii]
            ii = j2 - 1
            while ii >= 0:
                xp1[i] = x2[ii]
                i += 1
                yp1[i] = y2[ii]
                ii -= 1
            regions.append(("polygon", np.column_stack((xp1[0:i], yp1[0:i]))))
            return regions

        for i in range(j):
            xp1[i] = x[i]
            yp1[i] = y[i]
        if azi[0][0] - azi[0][1] > np.pi:
            azi[0][0] -= 2. * np.pi
        elif azi[0][1] - azi[0][0] > np.pi:
            azi[0][0] += 2. * np.pi
        if azi[0][0] < azi[0][1]:
            az = azi[0][1] - _D2R
            while az > azi[0][0]:
                si, co = np.sin(az), np.cos(az)
                xp1[i], yp1[i] = radius_size * si, radius_size * co
                i += 1
                az -= _D2R
        else:
            az = azi[0][1] + _D2R
            while az < azi[0][0]:
                si, co = np.sin(az), np.cos(az)
                xp1[i], yp1[i] = radius_size * si, radius_size * co
                i += 1
                az += _D2R
        regions.append(("polygon", np.column_stack((xp1[0:i], yp1[0:i]))))

        for i in range(j2):
            xp2[i] = x2[i]
            yp2[i] = y2[i]
        if azi[1][0] - azi[1][1] > np.pi:
            azi[1][0] -= 2. * np.pi
        elif azi[1][1] - azi[1][0] > np.pi:
            azi[1][0] += 2. * np.pi
        if azi[1][0] < azi[1][1]:
            az = azi[1][1] - _D2R
            while az > azi[1][0]:
                si, co = np.sin(az), np.cos(az)
                xp2[i] = radius_size * si
                i += 1
                yp2[i] = radius_size * co
                az -= _D2R
        else:
            az = azi[1][1] + _D2R
            while az < azi[1][0]:
                si, co = np.sin(az), np.cos(az)
                xp2[i] = radius_size * si
                i += 1
                yp2[i] = radius_size * co
                az += _D2R
        regions.append(("polygon", np.column_stack((xp2[0:i], yp2[0:i]))))
        return regions

    # n == 2
    for i in range(j3):
        xp1[i] = x3[i]
        yp1[i] = y3[i]
    for ii in range(j):
        xp1[i] = x[ii]
        i += 1
        yp1[i] = y[ii]
    if big_iso > 0:
        ii = j2 - 1
        while ii >= 0:
            xp1[i] = x2[ii]
            i += 1
            yp1[i] = y2[ii]
            ii -= 1
        regions.append(("polygon", np.column_stack((xp1[0:i], yp1[0:i]))))
        return regions

    if azi[2][0] - azi[0][1] > np.pi:
        azi[2][0] -= 2. * np.pi
    elif azi[0][1] - azi[2][0] > np.pi:
        azi[2][0] += 2. * np.pi
    if azi[2][0] < azi[0][1]:
        az = azi[0][1] - _D2R
        while az > azi[2][0]:
            si, co = np.sin(az), np.cos(az)
            xp1[i] = radius_size * si
            i += 1
            yp1[i] = radius_size * co
            az -= _D2R
    else:
        az = azi[0][1] + _D2R
        while az < azi[2][0]:
            si, co = np.sin(az), np.cos(az)
            xp1[i] = radius_size * si
            i += 1
            yp1[i] = radius_size * co
            az += _D2R
    regions.append(("polygon", np.column_stack((xp1[0:i], yp1[0:i]))))

    for i in range(j2):
        xp2[i] = x2[i]
        yp2[i] = y2[i]
    if azi[1][0] - azi[1][1] > np.pi:
        azi[1][0] -= 2. * np.pi
    elif azi[1][1] - azi[1][0] > np.pi:
        azi[1][0] += 2. * np.pi
    if azi[1][0] < azi[1][1]:
        az = azi[1][1] - _D2R
        while az > azi[1][0]:
            si, co = np.sin(az), np.cos(az)
            xp2[i] = radius_size * si
            i += 1
            yp2[i] = radius_size * co
            az -= _D2R
    else:
        az = azi[1][1] + _D2R
        while az < azi[1][0]:
            si, co = np.sin(az), np.cos(az)
            xp2[i] = radius_size * si
            i += 1
            yp2[i] = radius_size * co
            az += _D2R
    regions.append(("polygon", np.column_stack((xp2[0:i], yp2[0:i]))))
    return regions


def _first_motion_sign(M, takeoff, az, remove_isotropic=False):
    """Signo de gamma^T M gamma (radiación P) en la dirección (takeoff, az),
    con takeoff medido desde el nadir (0) y az desde el norte."""
    gamma = np.array([
        np.sin(takeoff) * np.cos(az),
        np.sin(takeoff) * np.sin(az),
        np.cos(takeoff),
    ])
    M_eval = M - (np.trace(M) / 3.0) * np.eye(3) if remove_isotropic else M
    value = gamma @ M_eval @ gamma
    return 1.0 if value > 0 else (-1.0 if value < 0 else 0.0)


def _polygon_centroid(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = np.sum(cross) / 2.0
    if np.fabs(area) < 1e-12:
        return float(np.mean(x)), float(np.mean(y))
    cx = np.sum((x + x1) * cross) / (6.0 * area)
    cy = np.sum((y + y1) * cross) / (6.0 * area)
    return float(cx), float(cy)


def _region_color(M, vertices, remove_isotropic=False):
    """'gray' (compresión) o 'white' (dilatación) según el signo de la
    radiación P en el centroide de la región."""
    if vertices is None:
        cx, cy = 0.0, 0.0
    else:
        cx, cy = _polygon_centroid(vertices)
    r = min(np.hypot(cx, cy), 1.0)
    takeoff = 2.0 * np.arcsin(r / np.sqrt(2.0))
    az = np.arctan2(cx, cy)
    sign = _first_motion_sign(M, takeoff, az, remove_isotropic=remove_isotropic)
    return "gray" if sign >= 0 else "white"


def _draw_regions(ax, regions, colors, radius=1.0):
    for zorder, ((kind, verts), color) in enumerate(zip(regions, colors), start=1):
        if kind == "disk":
            patch = Circle((0, 0), radius, facecolor=color, edgecolor='none', zorder=zorder)
        else:
            patch = Polygon(verts, closed=True, facecolor=color, edgecolor='none', zorder=zorder)
        ax.add_patch(patch)


def _style_beachball_axes(ax, title):
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')


def plot_beachball_set(M, mechanism_name="Fault Mechanism"):
    """
    Visualiza en subplots los patrones de radiación de ondas P, SV, SH y S-total.

    El panel de P ("Standard Beach Ball") se construye con la misma
    matemática que usa obspy.imaging.beachball (ejes principales T/N/P vía
    eigendescomposición, planos nodales analíticos para doble-cupla pura o
    barrido azimutal para tensores generales, proyección equal-area de
    Schmidt/Lambert), coloreado con la paleta propia de WaveLadruno
    (gris=compresión, blanco=dilatación). `M` se recibe en la convención
    North-East-Down de `moment_tensor_from_strike_dip_rake` y se convierte
    internamente a Up-South-East para aplicar esas fórmulas.
    """
    plt.close('all')  # Cerrar figuras previas

    def draw_subplot(ax, x, y, u, title):
        circle = Circle((0, 0), 1, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        if not np.all(np.isnan(u)):
            ax.contourf(x, y, u, levels=[-1e10, 0, 1e10], colors=['white', 'gray'], alpha=0.8)
            ax.contour(x, y, u, levels=[0], colors='red', linewidths=2)
        _style_beachball_axes(ax, title)

    def draw_p_panel(ax, title):
        mt_use = _ned_to_use(M)
        mt_norm = mt_use / np.linalg.norm(mt_use)
        T, N, P = _mt2axes(mt_norm)
        if _is_pure_dc(T, N, P):
            plane1 = _mt2plane(mt_norm)
            regions = [("polygon", v) for v in _dc_regions(plane1.strike, plane1.dip, plane1.rake)]
        else:
            regions = _mt_regions(T, N, P, plot_zerotrace=True)
        colors = [_region_color(M, verts, remove_isotropic=True) for _, verts in regions]
        _draw_regions(ax, regions, colors)
        ax.add_patch(Circle((0, 0), 1, facecolor='none', edgecolor='black', linewidth=2, zorder=10))
        _style_beachball_axes(ax, title)

    # === Mallado esférico ===
    n = 200
    theta, phi = np.linspace(0, np.pi, n), np.linspace(0, 2*np.pi, n)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    gx = np.sin(theta_grid) * np.cos(phi_grid)
    gy = np.sin(theta_grid) * np.sin(phi_grid)
    gz = np.cos(theta_grid)
    r_hat = np.array([gx, gy, gz])
    theta_hat = np.array([
        np.cos(theta_grid) * np.cos(phi_grid),
        np.cos(theta_grid) * np.sin(phi_grid),
        -np.sin(theta_grid)
    ])
    phi_hat = np.array([
        -np.sin(phi_grid),
        np.cos(phi_grid),
        np.zeros_like(phi_grid)
    ])

    # === Desplazamiento u = M · γ ===
    u = np.zeros((3,) + theta_grid.shape)
    for i in range(3):
        for j, g in enumerate([gx, gy, gz]):
            u[i] += M[i, j] * g

    # === Proyecciones ===
    u_SV = np.sum(u * theta_hat, axis=0)
    u_SH = np.sum(u * phi_hat, axis=0)
    u_S = np.sqrt(u_SV**2 + u_SH**2)

    # === Proyección equal-area de Schmidt/Lambert (hemisferio inferior) ===
    # r = sqrt(2)*sin(takeoff/2), x = r*sin(az), y = r*cos(az) — misma fórmula
    # que obspy.imaging.beachball.plot_mt para el hemisferio de abajo (Down>=0).
    lower = theta_grid <= np.pi / 2
    takeoff, az = theta_grid, phi_grid
    r_proj = np.sqrt(2) * np.sin(takeoff / 2)
    x = np.where(lower, r_proj * np.sin(az), np.nan)
    y = np.where(lower, r_proj * np.cos(az), np.nan)

    u_dict = {
        "SV-wave\n(Vertical Shear)":     np.where(lower, u_SV, np.nan),
        "SH-wave\n(Horizontal Shear)":   np.where(lower, u_SH, np.nan),
        "S-wave Total\n(|SV| + |SH|)":   np.where(lower, u_S, np.nan)
    }

    # === Crear figura ===
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    draw_p_panel(axes.flat[0], "P-wave\n(Standard Beach Ball)")
    for ax, (title, u_plot) in zip(axes.flat[1:], u_dict.items()):
        draw_subplot(ax, x, y, u_plot, title)

    # === Mostrar tensor
    info = "Moment Tensor M:\n" + "\n".join(
        f"M{i+1}{j+1} = {M[i,j]:.2f}" for i in range(3) for j in range(3) if abs(M[i,j]) > 1e-10
    )
    fig.text(0.02, 0.98, info, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='lightblue', alpha=0.8))

    plt.suptitle(f'{mechanism_name}\nRadiation Patterns: P, SV, SH, S',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
