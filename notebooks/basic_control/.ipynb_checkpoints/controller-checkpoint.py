import ipywidgets.widgets as widgets
from jetbot import Robot
import traitlets

def main(): 
    controller = widgets.Controller(index=1)
    display(controller)
    robot = Robot()
    left_link = traitlets.dlink((controller.axes[1], 'value'), (robot.left_motor, 'value'), transform=lambda x: -x)
    right_link = traitlets.dlink((controller.axes[3], 'value'), (robot.right_motor, 'value'), transform=lambda x: -x)

