"""
 file with utility for drawing 
 the function for drawing the sketch comes from 
 the original project made by google with small changes
"""
# libraries required for visualisation:
import os
import svgwrite
import numpy as np
import tensorflow as tf
from IPython.display import SVG, display
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import data_Manager
import math
from matplotlib import animation

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)


def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


def slerp(p0, p1, t):
    """Spherical interpolation."""
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
    """Linear interpolation."""
    return (1.0 - t) * p0 + t * p1


# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
    data = data_Manager.to_normal_strokes(data)
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x 
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i,0])/factor
        y = float(data[i,1])/factor
        lift_pen = data[i, 2]
        p += command+str(x)+","+str(y)+" "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
    dwg.save()
    display(SVG(dwg.tostring()))


"""
Function for animate drawing. 
taken from 
https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Strokes_QuickDraw.ipynb#scrollTo=0ABX6O4kYwYS
"""
def create_animation(drawing, fps = 30, idx = 0, lw = 5): 
  
  seq_length = 0 
  
  xmax = 0 
  ymax = 0 
  
  xmin = math.inf
  ymin = math.inf
  
  #retreive min,max and the length of the drawing  
  for k in range(0, len(drawing)):
    x = drawing[k][0]
    y = drawing[k][1]

    seq_length += len(x)
    xmax = max([max(x), xmax]) 
    ymax = max([max(y), ymax]) 
    
    xmin = min([min(x), xmin]) 
    ymin = min([min(y), ymin]) 
    
  i = 0 
  j = 0
  
  # First set up the figure, the axis, and the plot element we want to animate
  fig = plt.figure()
  ax = plt.axes(xlim=(xmax+lw, xmin-lw), ylim=(ymax+lw, ymin-lw))
  ax.set_facecolor("white")
  line, = ax.plot([], [], lw=lw)

  #remove the axis 
  ax.grid = False
  ax.set_xticks([])
  ax.set_yticks([])
  
  # initialization function: plot the background of each frame
  def init():
      line.set_data([], [])
      return line, 

  # animation function.  This is called sequentially
  def animate(frame):    
    nonlocal i, j, line
    x = drawing[i][0]
    y = drawing[i][1]
    line.set_data(x[0:j], y[0:j])
    
    if j >= len(x):
      i +=1
      j = 0 
      line, = ax.plot([], [], lw=lw)
      
    else:
      j += 1
    return line,
  
  # call the animator.  blit=True means only re-draw the parts that have changed.
  anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames= seq_length + len(drawing), blit=True)
  plt.close()
  
  # save the animation as an mp4.  
  anim.save(f'video.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])