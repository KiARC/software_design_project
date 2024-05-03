import datetime
import time
import sys

import cv2
import numpy
from numpy import sin, cos, tan, pi, arccos, arctan, arcsin, arctan2, sqrt
import argparse
import os.path
import matplotlib.pyplot as plt
import PIL.Image
import PIL.ExifTags
import xml.etree.ElementTree as ET


def get_sun_pos(day, hour, min, sec, lat, long):
    """
    Compute the position of the sun in the sky
    Source: https://gml.noaa.gov/grad/solcalc/solareqns.PDF with modifications
    based on the source code of https://gml.noaa.gov/grad/solcalc/azel.html
    The equation of time is taken from the corresponding Wikipedia page
    :param day: The day of the year, where January 1st is 0
    :param hour: hour in UTC
    :param min: minute in UTC
    :param sec: second in UTC
    :param lat: The latitude of the observer in degrees
    :param long: The longitude of the observer in degrees
    :return: (azimuth angle, zenith angle) in radians
    """
    gamma = 2*pi/365 * (day + (hour - 12)/24)

    eot_n = 2*pi / 365.24
    eot_a = eot_n*(day + 10)
    eot_b = eot_a + 2*0.0167*sin(eot_n*(day - 2))
    eot_c = (eot_a - arctan(tan(eot_b) / 0.91747714052))/pi

    # eqtime = 229.18 * (0.000075 + 0.001868*cos(gamma) - 0.032077*sin(gamma) - 0.014615*cos(2*gamma) - 0.040849*sin(2*gamma))
    eqtime = 720*(eot_c - round(eot_c))
    fy = day + hour/24
    decl = 0.006918 - 0.399912*cos(gamma) + 0.070257*sin(gamma) - 0.006758*cos(2*gamma) + 0.000907*sin(2*gamma) - 0.002697*cos(3*gamma) + 0.00148*sin(3*gamma)
    # decl = arcsin(0.39779 * cos(2*pi/365.24 * (fy+10) + 2*0.0167*sin(2*pi/365.24 * (fy-2))))
    ha = ((hour*60 + min + sec/60 + eqtime + 4*long) / 4 - 180) * pi/180
    #if ha < -pi: ha += 2*pi
    ha = numpy.where(ha < -pi, ha + 2*pi, ha)

    lat = pi/180 * lat
    zenith = arccos(sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(ha))
    azimuth = pi - arccos((sin(lat)*cos(zenith) - sin(decl)) / (cos(lat)*sin(zenith)))
    #if ha > 0: azimuth = 2*pi - azimuth
    azimuth = numpy.where(ha > 0, 2*pi - azimuth, azimuth)
    return azimuth, zenith


def sphere2xy(param, a, z):
    x = int(param["fullWidth"] / 360 * (a * 180/pi - param["heading"]) + param["img_width"]//2 + param["xoffs"])
    return (x + param["xstart"]) % param["fullWidth"] - param["xstart"], int(z / pi * param["fullHeight"] - param["ystart"] + param["yoffs"])


def extract_exif(file):
    img = PIL.Image.open(file)
    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in img.getexif().items()
        if k in PIL.ExifTags.TAGS
    }
    for n, v in img.applist:
        if v.startswith(b"http://ns.adobe.com/xap/1.0/"):
            xmp = ET.fromstring(v.split(b"\x00", 1)[1].decode())[0][0].items()
            for xk, xv in xmp:
                exif[xk.rsplit("}", 1)[1]] = xv
    for xk, xv in exif.items():
        print(xk.ljust(50), xv)
    return exif


def rot(angle):
    return numpy.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])


def get_azel(coords):
    """
    Given a point [x, y, z], return the azimuth angle, the zenith angle, and the magnitude. The y-axis points north
    and the x-axis points east. Azimuth angles are measured clockwise from north
    """
    x, y, z = coords
    return numpy.arctan2(x, y)%(2*pi), numpy.arctan2(numpy.linalg.norm([y, x]), z), numpy.linalg.norm([x, y, z])


def other_angle(R1, R2, a, b):
    """
    Given a point in space defined by R2 @ rot(b) @ R1 @ rot(a) @ [1,0,0],
    find another set of angles a' and b' such that:
    R2 @ rot(b') @ R1 @ rot(a') = R2 @ rot(b) @ R1 @ rot(a)
    """
    svec = rot(b) @ R1 @ rot(a) @ [1,0,0]
    # rot(b') @ R1 @ rot(a') @ [1,0,0] = svec
    # R1_31*cos(a') + R1_32*sin(a') = svec[2]
    # cos(arctan2(R1_32, R1_31))*cos(a') + sin(arctan2(R1_32, R1_31))*sin(a') = svec[2]
    # cos(a' - arctan2(R1_32, R1_31)) = svec[2]
    # a' - arctan2(R1_32, R1_31) = +- arccos(svec[2])
    aopt1 = arccos(svec[2]/numpy.linalg.norm(R1[2,:2]))
    aopt2 = arctan2(R1[2,1], R1[2,0])
    aopt = ((-aopt1 if numpy.isclose(aopt1-aopt2, a) else aopt1)-aopt2) % (2*pi)
    pred = R1 @ rot(aopt) @ numpy.array([1, 0, 0])
    bopt = arctan2(svec[1], svec[0]) - arctan2(pred[1], pred[0])
    return aopt, bopt % (2*pi)


def get_best_position(points, R1, R2, a=None, b=None):
    # We need to find a and b that maximize P = (R2 @ rot(b) @ R1 @ rot(a) @ [1,0,0]) . sum(points)
    #   P = (rot(b) @ R1 @ rot(a) @ [1,0,0]) . (R2.T @ sum(points))
    # If the chosen axes have "dead zones", they will surround the z-axis
    # Let s = (R2.T @ sum(points))
    # We want to get (rot(b) @ R1 @ rot(a) @ [1,0,0]) to coincide with s. Since they are unit vectors,
    # they must be equal
    #   rot(b) @ R1 @ rot(a) @ [1,0,0] = s
    #   rot(b) @ R1 @ [cos(a),sin(a),0] = s
    #   s_3 = R1_31*cos(a) + R1_32*sin(a)
    #   s3/sqrt((R1_31)**2 + (R1_32)**2) = sin(arctan(R1_31/R1_32) + a)
    ps = numpy.sum(points, axis=0)
    #print("Sum of vectors is", ps)
    s = R2.T @ ps
    s = s / numpy.linalg.norm(s)
    # Vertical range of the given coordinate system.
    # Measured in radians to the equator.
    # Equivalent to arccos([0,0,1] . (R1 @ [0,0,1]))
    rng = arccos(R1[2, 2])
    if b is not None:
        # P = (rot(b) @ R1 @ rot(a) @ [1,0,0]) . s
        # P = (rot(a) @ [1, 0, 0]) . (R1.T @ rot(b).T @ s)
        pred = R1.T @ rot(-b) @ s
        aopt = arctan2(pred[1], pred[0])
        bopt = b
        pass
    elif a is not None:
        pass
    else:
        if abs(arcsin(s[2])) <= rng:
            # It is possible for the solar panel to point directly at the optimal position
            # (rot(b) @ R1 @ rot(a) @ [1,0,0])[2] = s[2]
            # (R1 @ [cos(a), sin(a), 0])[2] = s[2]
            # cos(a)*R1[2, 0] + sin(a)*R1[2, 1] = s[2]
            aopt = a or arcsin(s[2]/numpy.linalg.norm(R1[2,:2])) - arctan(R1[2,0]/R1[2,1])
            #print(cos(aopt)*R1[2, 0] + sin(aopt)*R1[2, 1], s[2])
        else:
            # We need to find a value of a that maximizes the z-component R1 @ rot(a) @ [1, 0, 0]
            #   R1 @ [cos(a), sin(a), 0]
            #   z = R1_31 * cos(a) + R1_32 * sin(a)
            #   a = arctan(R1_32, R1_31)
            aopt = arctan2(R1[2,1], R1[2,0])
            aopt += (-1 if aopt > 0 else 1)*(pi if s[2] < 0 else 0)
        pred = R1 @ rot(aopt) @ numpy.array([1, 0, 0])
        #print(pred, s)
        bopt = arctan2(s[1], s[0]) - arctan2(pred[1], pred[0])
    return aopt, bopt, numpy.dot(rot(bopt) @ R1 @ rot(aopt) @ [1,0,0], s) * numpy.linalg.norm(ps)


def get_best_btrack(points, R1, R2, a=None):
    # a is the angle we are given, b_opt is the optimal b-angle
    # svec = R1 @ rot(a) @ [1, 0, 0]
    # Need to find maximum P = (R2 @ rot(b_opt) @ svec) . vc
    # P = (rot(a) @ svec) . (R2.T @ vc)
    # Maximum when az(rot(a) @ svec) = az(R2.T @ vc)
    # a + az(svec) = az(R2.T @ vc)
    # At this point, the angle between svec and R2.T @ vc is the difference in zenith angles
    # P_max = |vc| * cos(ze(svec) - ze(R2.T @ vc))
    rv = (R2.T @ points)
    srv = numpy.sum(rv[2])
    crv = numpy.sum(numpy.sqrt(rv[0]**2 + rv[1]**2))
    if a is not None:
        svec = R1 @ rot(a) @ numpy.array([1, 0, 0])
        return a, svec[2]*srv + numpy.sqrt(1-svec[2]**2)*crv
    else:
        # We need to find the best a such that P_max is maximized
        # This means we need to minimize the difference in zenith angles, or minimize
        # the difference in z-coordinates
        # svec[2] = R1_31 * cos(a) + R1_32 * sin(a)
        # Maximize: svec[2] * srv + numpy.sqrt(1 - svec[2] ** 2) * crv
        #   R1_31*srv * cos(a) + R1_32*srv * sin(a) + numpy.sqrt(1 - (R1_31 * cos(a) + R1_32 * sin(a))**2)*crv
        #   take a derivative
        #   solve for a
        a = arctan2(R1[2, 1], R1[2, 0]) + arccos(srv / (numpy.hypot(srv, crv)*numpy.hypot(R1[2, 1], R1[2, 0])))
        return get_best_btrack(points, R1, R2, a)
        pass


def print_power(p):
    print("    Total energy (J):               {} J/m^2/year".format(p))
    print("    Total energy (kWh):             {} kWh/m^2/year".format(p / 3600000))
    print("    Average power:                  {} W/m^2".format(p / (365 * 24 * 3600)))


def process_image(fobj, user_param):
    exif = extract_exif(fobj)
    param = {
        "fullHeight": int(exif["FullPanoHeightPixels"]),
        "fullWidth": int(exif["FullPanoWidthPixels"]),
        "ystart": int(exif["CroppedAreaTopPixels"]),
        "xstart": int(exif["CroppedAreaLeftPixels"]),
        "heading": float(exif["PoseHeadingDegrees"]),
        "xoffs": 0,
        "yoffs": 0
    }
    if "GPS Latitude" in exif and "GPS Longitude" in exif and not user_param.get("ignore_exif_coords", False):
        lat = exif["GPS Latitude"]
        long = exif["GPS Longitude"]
        lat = float(lat[:-2]) * (-1 if lat[-1] == "S" else 1)
        long = float(long[:-2]) * (-1 if long[-1] == "W" else 1)
        print("Extracted latitude and longitude", lat, long)
    else:
        if "latitude" not in user_param or "longitude" not in user_param:
            print("If GPS coordinates are not included in image EXIF data, "
                  "they must be provided with to -a and -o arguments")
            exit(1)
        lat = user_param["latitude"]
        long = user_param["longitude"]
    
    dt = user_param.get("dt", 15)  # Time between sample points in minutes
    p_len = int(24 * 60 // dt)  # Number of sample points per day
    positions = numpy.zeros((365 * p_len, 7))
    valid = []
    
    # Read the image and convert to grayscale
    fobj.seek(0)
    img_color = cv2.imdecode(numpy.asarray(bytearray(fobj.read()), dtype=numpy.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    param["img_height"], param["img_width"] = height, width

    mask = numpy.zeros((height+2, width+2), dtype=numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    selected_points = []
    img_temp = img.copy()

    def clicked2(event, x, y, flags, param):
        img, selected_points = param
        if event == cv2.EVENT_LBUTTONDOWN:
            if selected_points:
                cv2.line(img_temp, selected_points[-1], (x, y), (0, 0, 255), 5)
            selected_points.append((x, y))
            t0 = time.time()
            cv2.circle(img_temp, (x, y), 20, (0, 0, 255), -1)
            cv2.imshow("Panorama", img_temp)
            cv2.setMouseCallback("Panorama", clicked2, param)
            print("callback", time.time() - t0)
    

    if "selected_points" in user_param:
        pts = numpy.array(user_param["selected_points"]).astype(int)
    else:
        cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)
        cv2.imshow("Panorama", img)
        cv2.setMouseCallback("Panorama", clicked2, [img, selected_points])
        cv2.waitKey(0)
        pts = numpy.array(selected_points)
    
    start_imgprocess = time.time()
    xmin, xmax = numpy.min(pts[:, 0]), numpy.max(pts[:, 0])
    ymin, ymax = numpy.min(pts[:, 1]), numpy.max(pts[:, 1])
    colors = []
    # Sample 1000 random points and check each one if it is inside the shape the user selected.
    # If it is, store it for processing
    for i in range(1000):
        xi = numpy.random.randint(xmin, xmax)
        yi = numpy.random.randint(ymin, ymax)
        # Create a vertical line through (xi, yi)
        # The number of edges this line intersects must be odd on both sides
        inside = False
        top = 0
        bottom = 0
        for j in range(len(pts)):
            # collinear if (y1-y0)/(x1-x0) = (yi-y0)/(xi-x0)
            dx = pts[j, 0] - pts[j-1, 0]
            dy = pts[j, 1] - pts[j-1, 1]
            if pts[j-1, 0] <= xi <= pts[j, 0] or pts[j, 0] <= xi <= pts[j-1, 0]:
                if dy*(xi-pts[j-1, 0]) == dx*(yi - pts[j-1, 1]):
                    # point collinear with segment
                    inside = True
                    break
                # Vertical line cannot cross the segment, or we would have
                # caught it above
                if dx != 0:
                    # collinear if (y1-y0)/(x1-x0)*(xi-x0) = (yi+t-y0)
                    t = dy/dx * (xi - pts[j-1, 0]) + pts[j-1, 1] - yi
                    if t >= 0:
                        top += 1
                    else:
                        bottom += 1
        if inside or (top%2 and bottom%2):
            colors.append(img_color[yi, xi])
    colors = numpy.array(colors, dtype=float)
    print(colors.T @ colors)
    r = numpy.linalg.lstsq(colors, numpy.ones(len(colors)), rcond=None)
    rmse = numpy.sqrt(r[1][0]/len(colors)) * 100
    print("SSE: {}, RMSE: {}, condition number: {}".format(r[1], rmse, r[3][0] / r[3][-1]))
    r = r[0]
    diff = numpy.uint8(100*abs(img_color @ r - 1))
    if user_param.get("interactive", True):
        cv2.destroyAllWindows()
        cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
        cv2.imshow("diff", diff)
        cv2.waitKey(0)
    dil = cv2.dilate(diff, numpy.ones((100, 100)))
    thr = cv2.threshold(dil, numpy.uint8(rmse*10), 255, cv2.THRESH_BINARY_INV)[1]
    cv2.floodFill(thr, None, pts[0], 128)
    mask[1:-1, 1:-1] = numpy.where(thr == 128, 255, 0)
    print("Image processing took", time.time() - start_imgprocess)
    
    sun_pos = None
    
    def sun_callback(event, x, y, flags, param):
        global sun_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            img_temp = img.copy()
            cv2.circle(img_temp, (x, y), 20, (0, 0, 255), -1)
            sun_pos = (x, y)
            cv2.imshow("Select position of sun", img_temp)
            cv2.setMouseCallback("Select position of sun", sun_callback)
    
    # If complete time information is included, allow the user to select the position of the
    # sun to calibrate the compass data included in the image.
    offs = 0
    if "Date/Time Original" in exif and "Offset Time" in exif and user_param.get("interactive", True):
        d = datetime.datetime.strptime(exif["Date/Time Original"][:19], "%Y:%m:%d %H:%M:%S").timetuple()
        offs = exif["Offset Time"]
        offs = (int(offs[1:3]) + int(offs[4:6]) / 60) * (-1 if offs[0] == "-" else 1)
        print("Picture taken at", d, "offset", offs)
        target = get_sun_pos(d.tm_yday, d.tm_hour - offs, d.tm_min, d.tm_sec, lat, long)
        cv2.destroyAllWindows()
        cv2.namedWindow("Select position of sun", cv2.WINDOW_NORMAL)
        cv2.imshow("Select position of sun", img)
        cv2.setMouseCallback("Select position of sun", sun_callback)
        cv2.waitKey(0)
        if sun_pos is not None:
            print("position selected:", sun_pos)
            target = sphere2xy(param, *target)
            param["xoffs"] = sun_pos[0] - target[0]
            param["yoffs"] = sun_pos[1] - target[1]
        # cv2.circle(img, sphere2xy(param, *target), 200, (0, 255, 255), 30)
    
    # Draw the grid
    for i in range(0, 360, 10):
        direc = sphere2xy(param, i * numpy.pi / 180, i * numpy.pi / 180)
        cv2.line(img, (direc[0], 0), (direc[0], height), (0, 0, 128), 5)
        cv2.line(img, (0, direc[1]), (width, direc[1]), (0, 0, 128), 5)
    
    # Draw the lines for east, west, north, and south
    east = sphere2xy(param, pi / 2, 0)[0]
    cv2.line(img, (east, 0), (east, height), (0, 0, 255), 10)
    cv2.putText(img, "E", (east, height), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
    south = sphere2xy(param, pi, 0)[0]
    cv2.line(img, (south, 0), (south, height), (0, 0, 255), 10)
    cv2.putText(img, "S", (south, height), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
    west = sphere2xy(param, 3 * pi / 2, 0)[0]
    cv2.line(img, (west, 0), (west, height), (0, 0, 255), 10)
    cv2.putText(img, "W", (west, height), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
    north = sphere2xy(param, 0, 0)[0]
    cv2.line(img, (north, 0), (north, height), (0, 0, 255), 10)
    cv2.putText(img, "N", (north, height), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
    # Draw the horizon
    cv2.line(img, (0, param["fullHeight"] // 2 - param["ystart"]), (width, param["fullHeight"] // 2 - param["ystart"]),
             (0, 0, 255), 5)
    
    # Compute sun position for every time interval
    I_ss = 0
    I_cs = 0
    I_c = 0
    I_s = 0
    time_values = []
    for d in range(0, 365):
        # img_c = img.copy()
        for i in range(0, p_len):
            ho = i * dt
            h = ho - 60*offs
            a, z = get_sun_pos(d, h//60, h%60, 0, lat, long)
            positions[i+d*p_len, 0] = a
            positions[i+d*p_len, 1] = z
            positions[i+d*p_len, 2] = d
            positions[i+d*p_len, 3] = i
            positions[i+d*p_len, 4:] = 0
            time_values.append(datetime.datetime(1970, 1, 1, 0, 0, 0) + datetime.timedelta(days=d, minutes=ho))

    ld = 0
    # Sums for the current day
    d_ss = d_cs = d_c = d_n = 0
    # Stores optimal (az, el) angles for each day. Used for best-groups analysis
    days = []
    # Stores optimal vectors for each day. Used for best-groups analysis
    days_c = []
    # Stores the number sunlight hours each day
    days_n = []
    start_collectsun = time.time()
    for idx, (a, z, d, i, ins1, ins2, ins3) in enumerate(positions):
        pt = sphere2xy(param, a, z)
        if d != ld:
            #print("day {} has {}, {}, {}".format(ld, d_cs, d_ss, d_c))
            ld = d
            days_c.append([d_cs, d_ss, d_c])
            days.append(get_azel((d_cs, d_ss, d_c)))
            days_n.append(d_n*dt/60)
            d_ss = d_cs = d_c = d_n = 0
        if 0 <= pt[1] < height and 0 <= pt[0] < width and mask[pt[1]+1, pt[0]+1]:
            # Draw red dot for sampling points with sun
            #cv2.circle(img, pt, 10, (0, 0, 255), -1)
            insolation = 1353 * 0.7**((cos(z))**-0.678)
            positions[idx, 4] = -sin(a) * sin(z) * insolation
            positions[idx, 5] = cos(a) * sin(z) * insolation
            positions[idx, 6] = cos(z) * insolation
            valid.append([a, z, insolation])
            # dI_ss is negative because azimuth is clockwise
            dI_ss = -(dt * 60) * sin(a) * sin(z) * insolation
            dI_cs = (dt * 60) * cos(a) * sin(z) * insolation
            dI_c = (dt * 60) * cos(z) * insolation
            dI_s = (dt * 60) * sin(z) * insolation
            I_ss += dI_ss
            I_cs += dI_cs
            I_c += dI_c
            I_s += dI_s
            d_ss += dI_ss
            d_cs += dI_cs
            d_c += dI_c
            d_n += 1
    print("Collecting sun data took", time.time() - start_collectsun)
    
    valid = numpy.array(valid).transpose()
    valid_c = numpy.array([
        -sin(valid[1]) * sin(valid[0]) * valid[2],
        sin(valid[1]) * cos(valid[0]) * valid[2],  # negative because azimuth angles are cw
        cos(valid[1]) * valid[2]
    ])
    
    days = numpy.array(days).transpose()
    days_c = numpy.array(days_c)
    if user_param.get("interactive", True):
        plt.plot(days_n)
        plt.xlabel("day of the year")
        plt.ylabel("hours of sunlight")
        plt.show()
    
    # Load the coordinate system
    if "coord" not in user_param:
        # R1 = numpy.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        R1 = numpy.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        R2 = numpy.eye(3)
        # R2[1, 1] = -1
    
    else:
        R1, R2 = user_param["coord"]
    
    # @@@@@@@@@@@@@@@@@@@@@@
    # Best-groups analysis, assuming azimuth and zenith angles used
    start_bestgroups = time.time()
    cs = numpy.concatenate([[[0, 0, 0]], numpy.cumsum(days_c, axis=0)])
    a0 = user_param["zenith"] * pi/180 if "zenith" in user_param else None
    b0 = user_param["azimuth"] * pi/180 if "azimuth" in user_param else None
    npos = user_param.get("groups", 2)
    pos = [int(365 * (x + 0.5) / npos) for x in range(npos)]
    # pos = [90, 256]
    change = True
    for iter in range(20):
        totals = [get_best_position([cs[pos[n + 1]] - cs[pos[n]]], R1, R2, a=a0, b=b0) for n in range(npos - 1)] + \
                 [get_best_position([cs[pos[0]] + cs[-1] - cs[pos[-1]]], R1, R2, a=a0, b=b0)]
        vectors = [R2 @ rot(b) @ R1 @ rot(a) @ [1, 0, 0] for a, b, c in totals]
        if not change: break
        change = False
        for i in range(npos):
            rgrad = numpy.dot(vectors[(i - 1) % npos] - vectors[i], days_c[pos[i]])
            lgrad = numpy.dot(vectors[i] - vectors[(i - 1) % npos], days_c[pos[i] - 1])
            if lgrad < 0 < rgrad:
                # Moving to the right increases and moving to the left decreases. Therefore, move right
                pos[i] += 1
                change = True
            elif lgrad > 0 > rgrad:
                # Moving to the left increases and moving to the right decreases. Therefore, move left
                pos[i] -= 1
                change = True
    print("Best groups")
    print("    Steps:                          {}".format(pos))
    jan1 = datetime.datetime(datetime.datetime.now().year, 1, 1)
    for i, t in zip(pos, totals):
        ts = (jan1 + datetime.timedelta(i)).strftime("%Y/%m/%d")
        print("        Date and position:          {} -> a: {}, b: {}".format(ts, t[0]*180/pi, t[1]*180/pi))
    print_power(sum(t[2] for t in totals))
    print("Best-groups analysis took", time.time() - start_bestgroups)
    
    dn = numpy.arange(364)
    cl = numpy.concatenate([[0]*pos[0], *[[i]*(pos[i]-pos[i-1]) for i in range(1, npos)], [0]*(364-pos[-1])])
    a_val, b_val = [], []
    for a, b, c in totals:
        if a<0 or b<0: a, b = other_angle(R1, R2, a, b)
        pt = R2 @ rot(b) @ R1 @ rot(a) @ [1, 0, 0]
        az, ze, mag = get_azel(pt)
        a_val.append(ze)
        b_val.append(az)
    print(a_val, b_val)
    print("cl", cl)
    if user_param.get("interactive", True):
        plt.scatter(days[0], days[1], c=cl)
        plt.scatter(b_val, a_val, c=(numpy.arange(npos)+1)%npos, marker="x")
        plt.xlabel("azimuth angle (radians)")
        plt.ylabel("zenith angle (radians)")
        plt.show()
    
    # Draw the dots representing the times when the sun would be visible
    for a, z, d, i in positions[:, :4]:
        pt = sphere2xy(param, a, z)
        if 0 <= pt[1] < height and 0 <= pt[0] < width and mask[pt[1] + 1, pt[0] + 1]:
            r = 10
            cv2.circle(img, pt, r, (0, 0, 255), -1)
    
    # Draw the grid
    pts = []
    for i in range(0, 360, 10):
        p = []
        for j in range(0, 360, 10):
            pt = R2 @ rot(j * pi / 180) @ R1 @ rot(i * pi / 180) @ [1, 0, 0]
            pt = sphere2xy(param, *get_azel(pt)[:2])
            p.append(pt)
        pts.append(p)
    for i in range(36):
        for j in range(36):
            if 0 <= pts[i][j][0] < width and 0 <= pts[i][j][1] < height:
                if 0 <= pts[i - 1][j][0] < width and 0 <= pts[i - 1][j][1] < height:
                    cv2.line(img, pts[i][j], pts[i - 1][j], (0, 128, 0), 5)
                if 0 <= pts[i][j - 1][0] < width and 0 <= pts[i][j - 1][1] < height:
                    cv2.line(img, pts[i][j], pts[i][j - 1], (0, 160, 0), 5)
    
    # Calculations for a non-tracking panel, with some, none, or all of the angles specified
    results = {}
    if not user_param.get("track_azimuth", False) and not user_param.get("track_zenith", False):
        f_best_ze, f_best_az, f_energy = get_best_position(valid_c.T, R1, R2, a=a0, b=b0)
        f_energy = f_energy * 60 * dt
        print("Fixed panel:")
        print("    Best solar panel azimuth angle: {}".format(f_best_az * 180 / pi))
        print("    Best solar panel zenith angle:  {}".format(f_best_ze * 180 / pi))
        print("    Total energy (J):               {} J/m^2/year".format(f_energy))
        print("    Total energy (kWh):             {} kWh/m^2/year".format(f_energy / 3600000))
        print("    Average power:                  {} W/m^2".format(f_energy / (365 * 24 * 3600)))
        svec = (R2 @ rot(f_best_az) @ R1 @ rot(f_best_ze) @ [1, 0, 0])
        # print(svec)
        if user_param.get("interactive", True):
            plt.plot(time_values, positions[:, 4:] @ svec)
            plt.show()
        
        pt = R2 @ rot(f_best_az) @ R1 @ rot(f_best_ze) @ [1, 0, 0]
        # print("point is", pt, get_azel(pt))
        pt = sphere2xy(param, *get_azel(pt)[:2])
        cv2.circle(img, pt, 50, (0, 255, 0), -1)
        cv2.circle(img, (pt[0], -pt[1]), 50, (0, 255, 255), -1)
        cv2.line(img, (pt[0], 0), (pt[0], height), (0, 255, 0), 5)
        results["fixed"] = {
            "a": f_best_ze,
            "b": f_best_az,
            "energy": f_energy
        }
    
    # Calculations for a panel that tracks along the b (azimuth) axis
    h_best_ze, h_energy = get_best_btrack(valid_c, R1, R2, a=a0)
    h_energy = 60 * dt * h_energy
    print("b-axis tracking panel:")
    print("    Best solar panel zenith angle:  {}".format(h_best_ze * 180 / pi))
    print("    Total energy (J):               {} J/m^2/year".format(h_energy))
    print("    Total energy (kWh):             {} kWh/m^2/year".format(h_energy / 3600000))
    print("    Average power:                  {} W/m^2".format(h_energy / (365 * 24 * 3600)))
    results["btrack"] = {
        "a": h_best_ze,
        "energy": h_energy
    }
    
    pts = [sphere2xy(param, *get_azel(R2 @ rot(i * pi / 180) @ R1 @ rot(h_best_ze) @ [1, 0, 0])[:2]) for i in
           range(0, 360, 10)]
    for i in range(36):
        if (0 <= pts[i][0] < width and 0 <= pts[i][1] < height and 0 <= pts[i - 1][0] < width and
                0 <= pts[i - 1][1] < height):
            cv2.line(img, pts[i], pts[i - 1], (0, 255, 255), 5)
    
    print("Computed parameters:")
    print("    I_ss:                           {} J/m^2".format(I_ss))
    print("    I_cs:                           {} J/m^2".format(I_cs))
    print("    I_c:                            {} J/m^2".format(I_c))
    print("    I_s:                            {} J/m^2".format(I_s))
    results["parameters"] = {
        "svec": (valid_c.sum(axis=1)*dt*60).tolist()
    }
    
    cv2.imwrite("/tmp/final.jpg", img)
    if user_param.get("interactive", True):
        cv2.destroyAllWindows()
        cv2.namedWindow("Processing", cv2.WINDOW_NORMAL)
        cv2.imshow("Processing", img)
        cv2.waitKey(0)
    
    return results


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("file")
    parse.add_argument("-a", "--latitude", help="Latitude in degrees")
    parse.add_argument("-o", "--longitude", help="Longitude in degrees")
    parse.add_argument("-A", "--azimuth", default=None,
                       help="Azimuth angle of the panel. If not specified, the optimal angle is computed. Specify "
                       "'track' for a panel that tracks along this axis.")
    parse.add_argument("-Z", "--zenith", default=None,
                       help="Zenith angle of the panel. If not specified, the optimal angle is computed. Specify "
                       "'track' for a panel that tracks along this axis.")
    parse.add_argument("--coord", default=None,
                       help="Rotation matrices that specify the coordinate system. If not specified, azimuth and "
                            "zenith angles are used. Use this setting when working with panels that track along a "
                            "different axis or set of axes.")
    parse.add_argument("--groups", default=2, help="Number of groups to use for best-groups analysis")
    parse.add_argument("--dt", default=15, help="Time increment for checking sun position, in minutes")
    args = parse.parse_args()
    
    user_param = {
        "interactive": True
    }
    if args.latitude is not None: user_param["latitude"] = float(args.latitude)
    if args.longitude is not None: user_param["longitude"] = float(args.longitude)
    if args.azimuth is not None:
        if args.azimuth == "track": user_param["track_azimuth"] = True
        else: user_param["azimuth"] = float(args.azimuth)
    if args.zenith is not None:
        if args.zenith == "track": user_param["track_zenith"] = True
        else: user_param["zenith"] = float(args.zenith)
    if args.coord is not None:
        c = args.coord.split(";")
        R1 = numpy.array([float(x) for x in c[0].split(",")]).reshape((3, 3))
        if not numpy.allclose(R1 @ R1.T, numpy.eye(3)): print("R1 must be orthogonal")
        R2 = numpy.array([float(x) for x in c[1].split(",")]).reshape((3, 3))
        if not numpy.allclose(R2 @ R2.T, numpy.eye(3)): print("R2 must be orthogonal")
        user_param["coord"] = (R1, R2)
    if args.dt is not None: user_param["dt"] = float(args.dt)
    if args.groups is not None: user_param["groups"] = int(args.groups)
    with open(args.file, "rb") as f:
        process_image(f, user_param)
