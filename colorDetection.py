def image_rec(input_image):
  from colorthief import ColorThief
  import colorsys
  import cv2
  from urllib import request
  from flask import Flask, request
  from flask import send_file
  from flask_restx import Resource, Api
  from werkzeug.utils import secure_filename
  
  sum_list = [0] * 10
  ct = ColorThief(input_image)
  dominant_color = ct.get_color(quality = 1)

  (r, g, b) = (dominant_color[0] / 255, dominant_color[1] / 255, dominant_color[2] / 255)

  def image_com(image):
    result = 0
    imgs = []
    imgs.append(cv2.imread(image, cv2.COLOR_BGR2GRAY))
    imgs.append(cv2.imread(input_image, cv2.COLOR_BGR2GRAY))
    
    hists = []
    for img in imgs:
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
      cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
      hists.append(hist)
    
    methods = ['CORREL', 'CHISQR', 'INTERSECT', 'BHATTACHARYYA', 'EMD']
    for index, name in enumerate(methods):
      for i, histogram in enumerate(hists):
        ret = cv2.compareHist(hists[0], histogram, 4)            
        if index == 4:
          if i == 1:
            result = ret
    return result

  index = 0
  sum_list[0] = image_com("Acrylic_1.jpg") + image_com("Acrylic_2.jpg") + image_com("Acrylic_3.jpg")
  sum_list[1] = image_com("Cotton_1.jpg") + image_com("Cotton_2.jpg") + image_com("Cotton_3.jpg")
  sum_list[2] = image_com("Denim_1.jpg") + image_com("Denim_2.jpg") + image_com("Denim_3.jpg")
  sum_list[3] = image_com("Gimo_1.jpg") + image_com("Gimo_2.jpg") + image_com("Gimo_3.jpg")
  sum_list[4] = image_com("Leather_1.jpg") + image_com("Leather_2.jpg") + image_com("Leather_3.jpg")
  sum_list[5] = image_com("Linen_1.jpg") + image_com("Linen_2.jpg") + image_com("Linen_3.jpg")
  sum_list[6] = image_com("Nylon_1.jpg") + image_com("Nylon_2.jpg") + image_com("Nylon_3.jpg")
  sum_list[7] = image_com("Poly_1.jpg") + image_com("Poly_2.jpg") + image_com("Poly_3.jpg")
  sum_list[8] = image_com("Rayon_1.jpg") + image_com("Rayon_2.jpg") + image_com("Rayon_3.jpg")
  sum_list[9] = image_com("Wool_1.jpg") + image_com("Wool_2.jpg") + image_com("Wool_3.jpg")

  index = sum_list.index(min(sum_list))

  if dominant_color[2] - dominant_color[0] > 15:
    if dominant_color[2] - dominant_color[1] > 15:
      index = 2

  array = ["Acrylic", "Cotton", "Denim", "Gimo", "Leather", "Linen", "Nylon", "Poly", "Rayon", "Wool"]

  fabric_type = array[index]
      
  sort_list = sorted(sum_list)
  for x in sum_list:
    if x == sort_list[1]:
      sec_index = sum_list.index(x)

  if dominant_color[2] - dominant_color[0] > 15:
    if dominant_color[2] - dominant_color[1] > 15:
      sec_index = sum_list.index(min(sum_list))

  second_type = array[sec_index]
  
  def result_return():
    return dominant_color[0], dominant_color[1], dominant_color[2], fabric_type, second_type
  
  result_return()