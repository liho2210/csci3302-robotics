"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard, GPS

import math
from ikpy.chain import Chain
import heapq
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space



###############################################
### CONTROL
###############################################
color = 'yellow'
mode = 'auto'
final_goal = [5.55, 3.7]
start_goal = [1, 5]


###############################################
### INIT
###############################################
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")


# Odometry
pose_x     = 2
pose_y     = 5
#print(pose_x, pose_y, "test")
pose_theta = 0

vL = MAX_SPEED
vR = MAX_SPEED




lidar_sensor_readings = []
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # remove blocked sensor rays

def normalize_angle(theta):
    if theta > 0:
        while theta > math.pi:
            theta -= 2 * math.pi
    elif theta < 0:
        while theta < -1 * math.pi:
            theta += 2 * math.pi
    return theta

###############################################
### ARM INIT
###############################################
# arm setup
base_elements = ["TIAGo front arm_21477"]
base_elt_defaults = [0]
link_mask = [True,  True,  True,  True,
        True,  True,  True,  True,
        False, False, False]
arm_chain = Chain.from_urdf_file('tiago_arm.urdf',
                                 base_elements=base_elements,
                                 active_links_mask=link_mask)

arm_motors = []
for link in arm_chain.links[1:8]:
    motor = robot.getDevice(link.name)
    motor.setVelocity(motor.getMaxVelocity() / 4)
    position_sensor = motor.getPositionSensor()
    position_sensor.enable(timestep)
    arm_motors.append(motor)

left_finger = robot.getDevice('gripper_left_finger_joint')
right_finger = robot.getDevice('gripper_right_finger_joint')
left_finger.getPositionSensor().enable(timestep)
right_finger.getPositionSensor().enable(timestep)
finger_max = left_finger.getMaxPosition()

target_above = [0, 0.8, 0.3]
target_or = [0, 0, 10]
above_goal_config = arm_chain.inverse_kinematics(target_position=target_above,
                                          target_orientation=target_or,
                                          orientation_mode='Y')
object_goal_config = None

def grab_step(prev_arm, prev_finger, grab_mode, obj_dist, object_goal_config, conv_count):

    target_object = [-0.01, obj_dist, -0.25]

    # set target poses
    if grab_mode == 'above' or grab_mode == 'hold':
        goal_config = above_goal_config
    elif grab_mode == 'descend' or grab_mode == 'close':
        if object_goal_config is None:
            object_goal_config = arm_chain.inverse_kinematics(target_position=target_object,
                                          target_orientation=target_or,
                                          orientation_mode='Y')
        goal_config = object_goal_config

    # get current pose from motor positions
    arm_configs = [m.getPositionSensor().getValue() for m in arm_motors]
    full_chain_config = base_elt_defaults + arm_configs + [0, 0, 0]
    h_pose = arm_chain.forward_kinematics(full_chain_config)
    end_eff_pos = h_pose[:3, 3]
    end_eff_or = h_pose[:3, 0]

    # set positions
    if grab_mode == 'close':
        left_finger.setPosition(0)
        right_finger.setPosition(0)
        f_pos = left_finger.getPositionSensor().getValue()

        if np.isclose(prev_finger, f_pos, atol=1e-5):
            conv_count += 1
            if conv_count == 5:
                conv_count = 0
                grab_mode = 'hold'

    else:
        # returning to high point with the object
        if grab_mode == 'hold':
            left_finger.setPosition(0)
            right_finger.setPosition(0)
            max_squeeze = left_finger.getMaxVelocity()
            left_finger.setVelocity(max_squeeze)
            right_finger.setVelocity(max_squeeze)
        # moving to above or descending to the object
        else:
            left_finger.setPosition(finger_max)
            right_finger.setPosition(finger_max)
        f_pos = left_finger.getPositionSensor().getValue()

        for i, m in enumerate(arm_motors):
            m.setPosition(goal_config[i + 1])
            m.setVelocity(m.getMaxVelocity() / 4)

        # convergence check
        if np.all(np.isclose(prev_arm, h_pose, atol=1e-10)):
            if grab_mode == 'above':
                grab_mode = 'descend'
            elif grab_mode == 'descend':
                if np.all(np.isclose(prev_arm[:3, 3], h_pose[:3, 3], 1e-4)):
                    conv_count += 1
                    if conv_count == 5:
                        conv_count = 0
                        grab_mode = 'close'

    return h_pose, f_pos, grab_mode, object_goal_config, conv_count

grab_mode = 'above'
prev_arm = arm_chain.forward_kinematics([0]*len(arm_chain))
prev_finger = 0
conv_count = 0


###############################################
### A* INIT
###############################################
def e_dist(x1, x2):
    return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

def convert_to_graph(env_map, x_size, y_size):

    graph_dict = {}
    for x in range(x_size):
        for y in range(y_size):
            neighbors = {}
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if x + i in range(x_size) and y + j in range(y_size) and (i != 0 or j != 0):
                        if env_map[x + i][y + j] == 0:
                            neighbors[(x + i, y + j)] = np.sqrt(i**2 + j**2)
            graph_dict[(x, y)] = neighbors
    return graph_dict

def path(prev, goal):
    if goal not in prev:
        return []
    else:
        return path(prev, prev[goal]) + [goal]

class Frontier_PQ:

    def __init__(self, start, cost):
        self.start = start
        self.cost = cost
        self.states = {start: cost}
        self.q = [(cost, start)]
        heapq.heapify(self.q)

    def contains(self, node):
        for elt in self.q:
            if elt[1] == node:
                return True
        return False

    def add(self, state, cost):
        heapq.heappush(self.q, (cost, state))
        if(state not in self.states):
            self.states[state] = cost

    def pop(self):
        return heapq.heappop(self.q)

    def replace(self, state, cost):
        self.states[state] = cost

def astar_search(start, goal, state_graph, heuristic, return_cost = False):

    # initialize explored set, frontier queue, and previous dict to track path
    explored = []
    frontier = Frontier_PQ(start, 0)
    previous = {start: None}

    # goal check
    if(start == goal):
        previous[goal] = start
        return (path(previous, goal), 0) if return_cost else (path(previous, goal))

    # loop while frontier isn't empty
    while(frontier.q):
        node = frontier.pop()
        while node[1] in explored:
            node = frontier.pop()

        neighbors = state_graph[node[1]]

        for succ in neighbors:

            if((not frontier.contains(succ)) and succ not in explored):

                # cost_to_succ is g, the cost to get to the successor; succ score is f = g + h
                cost_to_succ = frontier.states[node[1]] + state_graph[node[1]][succ]
                succ_score = cost_to_succ + heuristic(succ, goal)
                frontier.add(succ, succ_score)

                if(cost_to_succ <= frontier.states[succ]):
                    previous[succ] = node[1]
                    frontier.replace(succ, cost_to_succ)

        explored.append(node[1])
        if(goal in explored):
            pathToGoal = path(previous, goal)
            return (pathToGoal, pathcost(pathToGoal, state_graph)) if return_cost else (pathToGoal)

    print(explored[-50:-1])
    return ["failure"], -1 if return_cost else ["failure"]

def path_planner(sg, start, end):
    '''
    :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
    :param start: A tuple of indices representing the start cell in the map
    :param end: A tuple of indices representing the end cell in the map
    :return: A list of tuples as a path from the given start to the given end in the given maze
    '''
    map_path = astar_search(start, end, sg, e_dist)
    print(map_path)
    world_path = []
    interval = max(int(len(map_path) / 20), 1)
    for i, elt in enumerate(map_path):
        if i % interval == 0:
            world_path.append((elt[0] * 7 / 350, elt[1] * 7 / 350))

    final = (map_path[-1][0] * 7 / 350, map_path[-1][1] * 7 / 350)
    if final not in world_path: world_path.append(final)

    return world_path
    

###############################################
### CV INIT
###############################################

width = 0.1

P_COEFFICIENT = 0.075

goal = True
dist = 0.55
focal = 420
obj_dist = 0
fix_dist = 1.2
stop_dist = 1.15

robot_move = False
robot_arm = False
navigate = True

slow_dist = 1.3

if (color == 'red'):
    # RED
    lower = np.array([118,190,50])
    upper = np.array([140,255,255])
elif (color == 'blue'):
    # BLUE
    lower = np.array([170, 190, 0])
    upper = np.array([190, 255, 255])
else:
    # YELLOW
    lower = np.array([50, 190, 0])
    upper = np.array([100, 255, 255])


def get_image_from_camera():
    """
    Take an image from the camera device and prepare it for OpenCV processing:
    - convert data type,
    - convert to RGB format (from BGRA), and
    - rotate & flip to match the actual image.
    """
    img = camera.getImageArray()
    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return cv2.flip(img, 1)
    


def get_distance_from_camera(knownWidth, focalLength, perWidth):
    
    # D = (W x F) / P; Distance = (Width in meters x Focal Length) / Width in Pixels
    return (knownWidth * focalLength) / perWidth
    

###############################################
### NAV INIT
###############################################
if mode == 'manual':
    map1 = np.zeros((350, 350)) # Replace None by a numpy 2D floating point array
    
# goals = [(0.5, 5),(0.5,0.5),(6.5, 1), (6.5, 3), (5.55, 4.23)]
goals = [final_goal]
#goals = []
goal_count = 0
angle_error_prop = 5

if mode == 'auto':
    # processed map should be in the same directory as the controller
    world_path = np.load('map_update.npy')
    env_map = np.transpose(world_path)
    state_graph = convert_to_graph(env_map, len(env_map), len(env_map[0]))
    
    world_size = 7
    map_size = len(env_map)
    
    # start and goal, in 7x7 world coords
    
    start_w = start_goal
    end_w = final_goal
    
    # Convert the start_w and end_W from webot's coordinate frame to map's
    start = (int(start_w[0] * map_size / world_size),
             int(start_w[1] * map_size / world_size))
    end = (int(end_w[0] * map_size / world_size),
             int(end_w[1] * map_size / world_size))

###############################################
### A* IMP
############################################### 
    
    # goals = path_planner(state_graph, start, end)
    # goals = goals[1:]
    
    
###############################################
### MAIN LOOP
###############################################  
while robot.step(timestep) != -1:
    

    if mode == 'manual':
        
###############################################
### MANUAL NAV IMP
############################################### 
        pose_y = gps.getValues()[2]
        pose_x = gps.getValues()[0]
    
        n = compass.getValues()
        rad = -((math.atan2(n[2], n[0])))
        pose_theta = rad
    
        lidar_sensor_readings = lidar.getRangeImage()
        lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    
        for i, rho in enumerate(lidar_sensor_readings):
            alpha = lidar_offsets[i]
    
            # if rho > LIDAR_SENSOR_MAX_RANGE:
                # rho = LIDAR_SENSOR_MAX_RANGE
                
            #print(rho)
            rx = math.sin(alpha)*rho
            ry = math.cos(alpha)*rho
    
            # wx =  rx + pose_x
            # wy =  -ry + pose_y
            wy =  (math.sin(pose_theta)*rx - math.cos(pose_theta)*ry) + pose_y
            wx =  (math.cos(pose_theta)*rx + math.sin(pose_theta)*ry) + pose_x
            #print(wx, wy)
    
            if rho < 0.5*LIDAR_SENSOR_MAX_RANGE:
                
                
                r = int(wx*50)
                c = int(wy*50)
                
                if r < 350 and c < 350:
                
                    map1[r][c] = min(map1[r][c]+0.005, 1)
                    g = map1[r][c]
                    #map1[r][c] = g
                    
                    
                    hex_color = int((g*256**2+g*256+g)*255)
                    if hex_color > 0xFFFFFF:
                        hex_color = 0xFFFFFF
                    
                    # You will eventually REPLACE the following 2 lines with a more robust version of map
                    # and gray drawing that has more levels than just 0 and 1.
                    if map1[r][c] > 0.5:
                        display.setColor(int(hex_color))
                        display.drawPixel(int(wx*50),int(wy*50))
    
        display.setColor(int(0xFF0000))
        display.drawPixel(int(pose_x*50),int(pose_y*50))
    

        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):

            np.save('map1', map1)

            print("Map file saved")
        elif key == ord('L'):
            map1 = np.load("map1.npy")
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
            
            
            
    else: # autonomous
        print(f'Goalcount: {goal_count}')
        
        if goal_count >= len(goals):
            
###############################################
### CV IMP
############################################### 

            if not robot_arm:
                # Find Object
                img = get_image_from_camera()
                mask = cv2.inRange(img, lower, upper)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # Find contour and calculate the distance from robot using width of contour
                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    marker = cv2.minAreaRect(largest_contour)
                    if  marker[1][0] > 0:
                        obj_dist = get_distance_from_camera(width, focal, marker[1][0])
                        print(obj_dist)
                    largest_contour_center = cv2.moments(largest_contour)
                    if largest_contour_center['m00'] > 0:
                        center_x = int(largest_contour_center['m10'] / largest_contour_center['m00'])
                    else:
                        center_x = 0
                    error = camera.getWidth() / 2 - center_x
                    # print(center_x, camera.getWidth()/2)
                    print(error)
                    vL = -error * P_COEFFICIENT
                    vR = error * P_COEFFICIENT

                    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('window', 600,600)
                    cv2.imshow('window', mask)

                    cv2.waitKey(1)
                    # print(error)
                    if error <= 4 and error >= -4 and obj_dist > slow_dist:
                        vL = 1.5
                        vR = 1.5
                    elif obj_dist <= slow_dist and error <= 1 and error >= -1:


                        if np.isclose(obj_dist, stop_dist, atol = 5e-2):
                            vL = 0
                            vR = 0
                            goal = False
                            robot_arm = True
                        elif obj_dist > stop_dist:
                            vL = 0.75
                            vR = 0.75
                        elif obj_dist < stop_dist:
                            vL = -0.75
                            vR = -0.75


                else:
                    vL = 0.35
                    vR = -0.35

            else:
###############################################
### ARM IMP
############################################### 
                vL = 0
                vR = 0

                prev_arm, prev_finger, grab_mode, object_goal_config, conv_count = grab_step(prev_arm,
                                                             prev_finger,
                                                             grab_mode,
                                                             obj_dist,
                                                             object_goal_config,
                                                             conv_count)

                print(f'Mode: {grab_mode}, arm: {prev_arm[:3, 3]}, finger: {prev_finger}, object: {obj_dist}')

                if grab_mode == 'done':
                    break
                    
                
        
        
                        
        else:            
            print(f'Goal: {goals[goal_count]}')

###############################################
### AUTO NAV IMP
############################################### 

            pose_y = gps.getValues()[2]
            pose_x = gps.getValues()[0]
        
            n = compass.getValues()
            rad = -((math.atan2(n[2], n[0])))
            pose_theta = rad
    
            x_dist = pose_x - goals[goal_count][0]
            y_dist = pose_y - goals[goal_count][1]
            
            #print(x_dist, y_dist)
        
            dist_error = math.sqrt(x_dist**2 + y_dist**2)
            rho = math.atan2(y_dist, x_dist)
            angle_error = normalize_angle(rho -(math.pi/2)- pose_theta)
            print(f'Angle: {angle_error}, dist: {dist_error}, rho: {rho}') 
                           
            if dist_error <= 0.1:
                goal_count += 1
                
                #if goal_count == 21:
                    
            

            angular_speed = angle_error * angle_error_prop
            robot_speed = MAX_SPEED_MS
            if dist_error <= 0.1:
                robot_speed = 0.5 * MAX_SPEED_MS
                
            #print(robot_speed, angular_speed)
                
            # Compute wheelspeeds
            vL = (robot_speed + (AXLE_LENGTH * angular_speed) / 2)
            vR = (robot_speed - (AXLE_LENGTH * angular_speed) / 2)

        
            # Normalize wheelspeed
            if vL > MAX_SPEED:
                vL = MAX_SPEED
            elif vL < -1 * MAX_SPEED:
                vL = -1 * MAX_SPEED
                
            if vR > MAX_SPEED:
                vR = MAX_SPEED
            elif vR < -1 * MAX_SPEED:
                vR = -1 * MAX_SPEED


    print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta)) #/3.1415*180))

    # Actuator commands
    print(f'Speeds: {vL}, {vR}')
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    #print(goals)