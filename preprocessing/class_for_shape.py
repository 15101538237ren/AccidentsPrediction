# -*- coding: utf-8 -*-
import math
EPS = 1e-6
class Vector2:
    X = 0.0 #lng
    Y = 0.0 #lat
    def __init__(self,lng,lat):
        self.X = lng
        self.Y = lat
class Rect:
    #定义基本属性
    LeftTop = Vector2(0.0, 0.0) #(min_lng, max_lat)
    RightBottom = Vector2(0.0, 0.0) #(max_lng, min_lat)

    def __init__(self,left_top,right_bottom):
        self.LeftTop = left_top
        self.RightBottom = right_bottom
    def Contains(self,point):
        if (self.LeftTop.X <= point.X <= self.RightBottom.X) and (self.RightBottom.Y <= point.Y <= self.LeftTop.Y):
            return True
        else:
            return False

#检查线段和矩形是否相交
def CheckRectLine(start, end, rect):
    result = False
    if (rect.Contains(start) or rect.Contains(end)):
        result = True
    else:
        result |= CheckRectLineH(start, end, rect.LeftTop.Y, rect.LeftTop.X, rect.RightBottom.X)
        result |= CheckRectLineH(start, end, rect.RightBottom.Y, rect.LeftTop.X, rect.RightBottom.X)
        result |= CheckRectLineV(start, end, rect.LeftTop.X, rect.LeftTop.Y, rect.RightBottom.Y)
        result |= CheckRectLineV(start, end, rect.RightBottom.X, rect.LeftTop.Y, rect.RightBottom.Y)
    return result

#水平方向上的检测线段与矩形是否相交
def CheckRectLineH(start, end, y0, x1, x2):
    #直线在点的上方
    if ((y0 < start.Y) and (y0 < end.Y)):
        return False
    #直线在点的下方
    if ((y0 > start.Y) and (y0 > end.Y)):
        return False
    #水平直线
    if (math.fabs(start.Y - end.Y) < EPS):
        #水平直线与点处于同一水平。
        if (math.fabs(y0 - start.Y) < EPS):
            #直线在点的左边
            if ((start.X < x1) and (end.X < x1)):
                return False
            #直线在x2垂直线右边
            if ((start.X > x2) and (end.X > x2)):
                return False
            #直线的部分或者全部处于点与x2垂直线之间
            return True
        else:#水平直线与点不处于同一水平。
            return False
    #斜线
    x = (end.X - start.X) * (y0 - start.Y) / (end.Y - start.Y) + start.X
    return ((x >= x1) and (x <= x2))

#垂直方向上的检测线段与矩形是否相交
def CheckRectLineV(start, end, x0, y1, y2):
    if ((x0 < start.X) and (x0 < end.X)):
        return False
    if ((x0 > start.X) and (x0 > end.X)):
        return False
    if (math.fabs(start.X - end.X) < EPS):
        if (math.fabs(x0 - start.X) < EPS):
            if ((start.Y < y1) and (end.Y < y1)):
                return False
            if ((start.Y > y2) and (end.Y > y2)):
                return False
            return True
        else:
            return False
    y = (end.Y - start.Y) * (x0 - start.X) / (end.X - start.X) + start.Y
    return ((y >= y2) and (y <= y1))