"""Picrawler robot control utilities.

This module provides the `Picrawler` robot class built on top of
`robot_hat.Robot`. It contains conversions between Cartesian coordinates
and joint angles, gait definitions and helpers for calibration.

Documentation style: concise docstrings for classes and public methods,
inline comments for non-obvious math and control logic.
"""

from robot_hat import Robot, utils

import time
import math

class Picrawler(Robot):
    """High-level controller for the Picrawler hexapod robot.

    Extends `robot_hat.Robot` to provide coordinate -> servo-angle
    transformations, gait sequences (encapsulated in `MoveList`) and
    convenience helpers for calibration and actions.

    Attributes:
        A, B, C: physical linkage lengths used for forward/inverse kinematics.
        OFFSET_FILE: path to persisted servo offsets.
        PIN_LIST: default servo pin mapping.
    """
    A = 48
    B = 78
    C = 33
    OFFSET_FILE = '/opt/picrawler/picrawler.config'
    PIN_LIST = [9, 10, 11, 3, 4, 5, 0, 1, 2, 6, 7, 8]

    def __init__(self, pin_list=PIN_LIST, init_angles=None):  
        """Initialize the Picrawler robot controller.

        Resets the MCU, initializes the base `Robot` and prepares internal
        gait and coordinate state.
        """

        # Ensure MCU is in a known state before configuring servos
        utils.reset_mcu()
        time.sleep(0.2)

        super().__init__(pin_list, db=self.OFFSET_FILE, name='picrawler', init_angles=init_angles)

        # Predefined gait/motion lists
        self.move_list = self.MoveList()
        # User-extensible additional actions
        self.move_list_add = {
            'my action': None
        }

        # Named step sequences exposed to callers
        self.step_list = {
            "stand": self.move_list['stand'],
            "sit": self.move_list['sit'],
        }

        # internal state used by gaits
        self.stand_position = 0
        # direction multipliers for each servo to flip orientation where required
        self.direction = [
            1,1,-1,
            1,1,1,
            1,1,-1,
            1,1,1,
        ]

        # current Cartesian coordinates for the four legs
        self.current_coord = [[60, 0, -30], [60, 0, -30], [60, 0, -30], [60, 0, -30]]
        # temporary coordinate buffer used during computations
        self.coord_temp = [[60, 0, -30], [60, 0, -30], [60, 0, -30], [60, 0, -30]]

    def coord2polar(self, coord):
        """Convert a Cartesian `coord` [x,y,z] to leg joint angles.

        Performs inverse-kinematics for a single leg and returns the
        angles [alpha, beta, gamma] in degrees. The routine clamps the
        input vector to avoid unreachable positions and ensures numeric
        stability.
        """

        x, y, z = coord

        # distance from origin to target point
        L = math.sqrt(x**2 + y**2 + z**2)
        if L == 0:
            # avoid division by zero
            L = 0.1

        # Clamp the vector so it lies within the manipulator's reach
        if L < self.C:
            temp = self.C / L
            x = temp * x
            y = temp * y
            z = temp * z
        elif L > (self.A + self.B + self.C):
            temp = (self.A + self.B + self.C) / L
            x = temp * x
            y = temp * y
            z = temp * z

        # Save adjusted coordinate to the temp buffer
        self.coord_temp.append([x, y, z])

        # compute intermediate geometry
        w = math.sqrt(x**2 + y**2)
        v = w - self.C
        u = math.sqrt(z**2 + v**2)
        # clamp leg extension
        u = max(30, min(91.58, u))

        # law of cosines to get the knee angle (beta)
        cos_angle1 = (self.B**2 + self.A**2 - u**2) / (2 * self.B * self.A)
        beta = math.acos(cos_angle1)

        # compute shoulder/hip angle components
        angle1 = math.atan2(z, v)
        angle2 = math.acos((self.A**2 + u**2 - self.B**2) / (2 * self.A * u))
        alpha = angle2 + angle1

        # yaw around the body (gamma)
        gamma = math.atan2(y, x)

        # convert to degrees and apply offsets used by the robot's kinematic convention
        alpha = 90 - alpha / math.pi * 180
        beta = beta / math.pi * 180 - 90
        gamma = -(gamma / math.pi * 180 - 45)

        return [round(alpha, 4), round(beta, 4), round(gamma, 4)]

    def polar2coord(self, angles):
        """Convert joint `angles` [alpha,beta,gamma] (degrees) to Cartesian coords.

        This is the forward kinematics counterpart to `coord2polar`.
        """

        alpha, beta, gamma = angles

        # compute effective link length L1 using the law of cosines
        L1 = math.sqrt(self.A**2 + self.B**2 - 2 * self.A * self.B * math.cos((90 + alpha) / 180 * math.pi))
        angle = math.acos((self.A**2 + L1**2 - self.B**2) / (2 * self.A * L1)) * 180 / math.pi
        angle = 90 - beta - angle
        L = L1 * math.cos(angle * math.pi / 180) + self.C

        # project into body coordinate frame
        x = L * math.sin((45 + gamma) * math.pi / 180)
        y = L * math.cos((45 + gamma) * math.pi / 180)
        z = L1 * math.sin(angle * math.pi / 180)

        return [round(x, 4), round(y, 4), round(z, 4)]

    def limit(self,min,max,x):
        """Clamp `x` to the inclusive range [`min`, `max`]."""
        if x > max:
            return max
        elif x < min:
            return min
        else:
            return x

    def limit_angle(self,angles):
        """Ensure each angle is within the robot's allowed ranges.

        Returns a tuple `(limit_flag, [alpha,beta,gamma])` where `limit_flag`
        is True if any input value was clamped.
        """
        alpha, beta, gamma = angles
        limit_flag = False

        # clamp each joint to its safe operating range
        temp = self.limit(-90, 90, alpha)
        if temp != alpha:
            alpha = temp
            limit_flag = True

        temp = self.limit(-10, 90, beta)
        if temp != beta:
            beta = temp
            limit_flag = True

        temp = self.limit(-60, 60, gamma)
        if temp != gamma:
            gamma = temp
            limit_flag = True

        return limit_flag, [alpha, beta, gamma]

    def do_action(self, motion_name, step=1, speed=50):
        """Execute a named motion sequence `motion_name`.

        Looks up the motion in the built-in `move_list` first; if not
        found, tries `move_list_add` for user-defined sequences.
        """
        try:
            for _ in range(step):  # repeat `step` times
                # propagate stand position to the move list so properties can branch on it
                self.move_list.stand_position = self.stand_position
                if motion_name in ["forward", "backward", "turn left", "turn right", "turn left angle", "turn right angle"]:
                    # toggle stance between gaits when performing directional moves
                    self.stand_position = self.stand_position + 1 & 1
                action = self.move_list[motion_name]
                for _step in action:  # iterate over micro-steps in the gait
                    self.do_step(_step, speed=speed)
        except AttributeError:
            try:
                # fallback to additional custom actions provided by the user
                for _ in range(step):
                    action_add = self.move_list_add[motion_name]
                    for _step in action_add:
                        self.do_step(_step, speed=speed)
            except KeyError:
                print("No such action")

    def set_angle(self, angles_list, speed=50, israise=False):
        """Apply a list of joint `angles_list` to the servos.

        Each entry in `angles_list` is an [alpha,beta,gamma] triple for a
        leg. Values outside the safe range are either clamped or, if
        `israise` is True, cause an exception.
        """
        translate_list = []
        results = []
        for angles in angles_list:
            result, angles = self.limit_angle(angles)
            translate_list += angles
            results.append(result)

        if True in results:
            if israise:
                raise ValueError('\033[1;35mCoordinates out of controllable range.\033[0m')
            else:
                try:
                    # recalc current Cartesian coordinates from clamped angles
                    coords = []
                    for i in range(4):
                        coords.append(self.polar2coord([translate_list[i * 3], translate_list[i * 3 + 1], translate_list[i * 3 + 2]]))
                    self.current_coord = list.copy(coords)
                except Exception as e:
                    print('re : %s' % e)
        else:
            # accept the temporary coordinates computed earlier
            self.current_coord = list.copy(self.coord_temp)

        # send flattened angle list to low-level servo mover
        self.servo_move(translate_list, speed)
        return list.copy(translate_list)

    def do_step(self, _step, speed=50, israise=False):
        """Execute a single step description `_step`.

        `_step` may be a string key referencing a named gait in
        `self.step_list` or an explicit list of leg Cartesian coordinates.
        Each coordinate is converted to joint angles and applied.
        """
        if isinstance(_step, str):
            if _step in self.step_list.keys():
                for one_step in self.step_list[_step]:
                    angles_temp = []
                    for coord in one_step:  # each servo motion
                        alpha, beta, gamma = self.coord2polar(coord)
                        # some code paths expect [beta, alpha, gamma] ordering
                        angles_temp.append([beta, alpha, gamma])
                    self.coord_temp = list.copy(one_step)
                    self.set_angle(angles_temp, speed, israise)
            else:
                print("The name of gait is not in the default gait dictionary")
        elif isinstance(_step, list):
            angles_temp = []
            for coord in _step:  # each servo motion
                alpha, beta, gamma = self.coord2polar(coord)
                angles_temp.append([beta, alpha, gamma])
            self.coord_temp = list.copy(_step)
            self.set_angle(angles_temp, speed, israise)
        else:
            print("The \"_step\" parameter is wrong.")
            return


    def current_step_all_leg_angle(self):
        """Return a copy of the current servo positions (angles)."""
        return list.copy(self.servo_positions)

    def add_action(self,action_name, action_list):
        """Register a user-defined action sequence under `action_name`."""
        self.move_list_add[action_name] = action_list


    def cali_helper_web(self, leg, pos, enter):
        """Helper used by the web calibration UI to nudge leg `leg`.

        `pos` is one of 'up','down','left','right','high','low'. When
        `enter` is set, the computed offset is saved to persistent
        storage via `set_offset`.
        """
        step = 0.2
        cali_position = []
        cali_coord = [[60, 0, -30], [60, 0, -30], [60, 0, -30], [60, 0, -30]]

        for coord in cali_coord:  # each servo motion
            alpha, beta, gamma = self.coord2polar(coord)
            cali_position += [beta, alpha, gamma]

        cali_position = [cali_position[i] + self.offset[i] for i in range(12)]

        positive_list = [
            [1, -1, -1, 1, 1, -1],
            [1, -1, 1, -1, 1, -1],
            [-1, 1, 1, -1, 1, -1],
            [-1, 1, -1, 1, 1, -1],
        ]

        offset = list.copy(self.offset)
        leg = leg - 1
        if pos == 'up':
            self.current_coord[leg][1] += step * positive_list[leg][0]
        elif pos == 'down':
            self.current_coord[leg][1] += step * positive_list[leg][1]
        elif pos == 'left':
            self.current_coord[leg][0] += step * positive_list[leg][2]
        elif pos == 'right':
            self.current_coord[leg][0] += step * positive_list[leg][3]
        elif pos == 'high':
            self.current_coord[leg][2] += step * positive_list[leg][4]
        elif pos == 'low':
            self.current_coord[leg][2] += step * positive_list[leg][5]

        # enforce workspace limits for each coordinate component
        for coord in self.current_coord:
            coord[0] = max(40, min(80, coord[0]))
            coord[1] = max(-20, min(20, coord[1]))
            coord[2] = max(-50, min(-10, coord[2]))
        self.do_step(self.current_coord, speed=100)
        current_position = list.copy(self.do_step(self.current_coord, speed=100))
        if enter == 1:
            tmp = [current_position[i] - cali_position[i] + offset[i] for i in range(len(current_position))]
            offset[leg * 3:(leg + 1) * 3] = tmp[leg * 3:(leg + 1) * 3]
            self.current_coord[leg] = [60, 0, -30]
            self.set_offset(offset)
            self.do_step(self.current_coord, speed=100)


    class MoveList(dict):
        
        LENGTH_SIDE = 77
        X_DEFAULT = 45
        X_TURN = 70
        X_START = 0
        Y_DEFAULT = 45
        Y_TURN = 130
        Y_WAVE =120
        Y_START = 0 
        Z_DEFAULT = -50
        Z_UP = -30
        Z_WAVE = 60
        Z_TURN = -40
        Z_PUSH = -76
         
        # temp length
        TEMP_A = math.sqrt(pow(2 * X_DEFAULT + LENGTH_SIDE, 2) + pow(Y_DEFAULT, 2))
        TEMP_B = 2 * (Y_START + Y_DEFAULT) + LENGTH_SIDE
        TEMP_C = math.sqrt(pow(2 * X_DEFAULT + LENGTH_SIDE, 2) + pow(2 * Y_START + Y_DEFAULT + LENGTH_SIDE, 2))
        TEMP_ALPHA = math.acos((pow(TEMP_A, 2) + pow(TEMP_B, 2) - pow(TEMP_C, 2)) / 2 / TEMP_A / TEMP_B)
        # site for turn
        TURN_X1 = (TEMP_A - LENGTH_SIDE) / 2
        TURN_Y1 = Y_START + Y_DEFAULT / 2
        TURN_X0 = TURN_X1 - TEMP_B * math.cos(TEMP_ALPHA)
        TURN_Y0 = TEMP_B * math.sin(TEMP_ALPHA) - TURN_Y1 - LENGTH_SIDE

        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)
            self.z_current = self.Z_UP
            self.stand_position = 0
            self.recovery_step = []
            self.ready_state = 0
            self.angle = 30
   
        def __getitem__(self, item):
            return eval("self.%s"%item.replace(" ", "_"))
        
        def turn_angle_coord(self, angle):
            a = math.atan(self.Y_DEFAULT/(self.X_DEFAULT+self.LENGTH_SIDE/2))
            angle1 = a/math.pi*180
            r1 = math.sqrt(pow(self.Y_DEFAULT,2)+ pow(self.X_DEFAULT+ self.LENGTH_SIDE/2, 2))
            x1 = r1* math.cos((angle1-angle)* math.pi/180)- self.LENGTH_SIDE/2
            y1 = r1* math.sin((angle1-angle)* math.pi/180)
            # print(x1,y1)
            
            x2 = (self.X_DEFAULT+ self.LENGTH_SIDE/2)* math.cos(angle*math.pi/180)- self.LENGTH_SIDE/2
            y2 = (self.X_DEFAULT+ self.LENGTH_SIDE/2)* math.sin(angle*math.pi/180)
            # print(x2,y2)
            
            b = math.atan((self.X_DEFAULT+self.LENGTH_SIDE/2)/(self.Y_DEFAULT+ self.LENGTH_SIDE))
            angle2 = b/math.pi*180
            r2 = math.sqrt(pow(self.X_DEFAULT+ self.LENGTH_SIDE/2, 2)+ pow(self.Y_DEFAULT+ self.LENGTH_SIDE,2))
            x3 = r2*math.sin((angle2-angle)* math.pi/180) - self.LENGTH_SIDE/2
            y3 = r2*math.cos((angle2-angle)*math.pi/180)- self.LENGTH_SIDE

            x3 += 10
            # print(x3,y3)
            return [x1,y1,x2,y2,x3,y3]
        
        # 装饰器封装函数,判断是否站立
        def check_stand(func):
            def wrapper(self):
                _action = []
                if not self.is_stand():
                    _action += self.stand
                _action += func(self)
                return _action
            return wrapper
        
        # 装饰器封装函数，装饰器简化步态的0，1两种状态转化，状态0为2，3脚y轴为0，状态1为1，4脚y轴为0 mode为2种转化方式，mode0为1，2交换3，4交换，mode1为1，3交换2，4交换
        def normal_action(mode):
            def wrapper1(func):
                def wrapper2(self):
                    _action = []
                    if self.stand_position == 0:
                        _action += func(self)
                    else:
                        temp = func(self)
                        new_step = []
                        for step in temp:
                            if mode == 0:
                                new_step = [step[1], step[0], step[3], step[2]]
                            elif mode == 1:
                                new_step = [step[2], step[3], step[0], step[1]]
                            _action += [new_step]
                    return _action
                return wrapper2
            return wrapper1
        
        @property
        @normal_action(0)
        def sit(self):
            self.z_current = self.Z_UP
            return [[
                [self.X_DEFAULT,self.Y_DEFAULT,self.z_current],
                [self.X_TURN,self.Y_START,self.z_current],
                [self.X_TURN,self.Y_START,self.z_current],
                [self.X_DEFAULT,self.Y_DEFAULT,self.z_current],
            ]]
            

        @property
        @normal_action(0)
        def stand(self):
            _stand = []
            if self.ready_state ==  0:
                _stand += self.ready
            self.z_current = self.Z_DEFAULT
            _stand += [[
                [self.X_DEFAULT,self.Y_DEFAULT,self.z_current],
                [self.X_DEFAULT,self.Y_START,self.z_current],
                [self.X_DEFAULT,self.Y_START,self.z_current],
                [self.X_DEFAULT,self.Y_DEFAULT,self.z_current],
            ]]
            return _stand
           
        
        @property
        def ready(self):
            _ready = [[
                [self.X_DEFAULT,self.Y_DEFAULT,self.z_current],
                [self.X_TURN,self.Y_START,self.z_current],
                [self.X_TURN,self.Y_START,self.z_current],
                [self.X_DEFAULT,self.Y_DEFAULT,self.z_current],
            ]]
            self.ready_state = 1
            return _ready
          

        def is_sit(self):
            return self.z_current == self.Z_UP
            
        def is_stand(self):
            tmp = self.z_current == self.Z_DEFAULT
            # print("is stand? %s"%tmp)
            return tmp
        
        @property
        @check_stand
        @normal_action(0)
        def forward(self):
            return [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_TURN, self.Y_START,self.Z_UP],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT*2,self.Z_UP],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT*2,self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT,self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT*2, self.z_current]],
                
                [[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT,self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT*2, self.Z_UP]],
                [[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT,self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_TURN, self.Y_START, self.Z_UP]],
                [[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT,self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current]],
            ]
        
        @property
        @check_stand
        @normal_action(0)
        def backward(self):
            return [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START,self.z_current],[self.X_TURN, self.Y_START, self.Z_UP],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START,self.z_current],[self.X_DEFAULT, self.Y_DEFAULT*2, self.Z_UP],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START,self.z_current],[self.X_DEFAULT, self.Y_DEFAULT*2, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT*2, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT,self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT*2, self.Z_UP],[self.X_DEFAULT, self.Y_DEFAULT,self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current]],
                [[self.X_TURN, self.Y_START, self.Z_UP],[self.X_DEFAULT, self.Y_DEFAULT,self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current]],
                [[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT,self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current]],
            ]
        
       
        @property
        @check_stand
        @normal_action(1)
        def turn_left(self):
            return [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START,self.z_current],[self.X_TURN, self.Y_START, self.Z_UP],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.TURN_X1, self.TURN_Y1, self.z_current],[self.TURN_X1, self.TURN_Y1, self.z_current],[self.TURN_X0, self.TURN_Y0, self.Z_UP],[self.TURN_X0, self.TURN_Y0, self.z_current]],
                [[self.TURN_X1, self.TURN_Y1, self.z_current],[self.TURN_X1, self.TURN_Y1, self.z_current],[self.TURN_X0, self.TURN_Y0, self.z_current],[self.TURN_X0, self.TURN_Y0, self.z_current]],
                
                [[self.TURN_X1, self.TURN_Y1, self.z_current],[self.TURN_X1, self.TURN_Y1, self.z_current],[self.TURN_X0, self.TURN_Y0, self.z_current],[self.TURN_X0, self.TURN_Y0, self.Z_UP]],
                [[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_TURN, self.Y_START, self.Z_UP]],
                [[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current]],
            ]

        @property
        @check_stand
        @normal_action(1)
        def turn_right(self):
            return [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_TURN, self.Y_START,self.Z_UP],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.TURN_X0, self.TURN_Y0, self.z_current],[self.TURN_X0, self.TURN_Y0, self.Z_UP],[self.TURN_X1, self.TURN_Y1, self.z_current],[self.TURN_X1, self.TURN_X1, self.z_current]],
                [[self.TURN_X0, self.TURN_Y0, self.z_current],[self.TURN_X0, self.TURN_Y0, self.z_current],[self.TURN_X1, self.TURN_Y1, self.z_current],[self.TURN_X1, self.TURN_X1, self.z_current]],
                [[self.TURN_X0, self.TURN_Y0, self.Z_UP],[self.TURN_X0, self.TURN_Y0, self.z_current],[self.TURN_X1, self.TURN_Y1, self.z_current],[self.TURN_X1, self.TURN_X1, self.z_current]],
                [[self.X_TURN, self.Y_START, self.Z_UP],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current]],
                [[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current]],
                
            ]
        
        @property
        def push_up(self):
            _push_up = []
            if not self.is_sit():
                _push_up += self.sit
            _push_up += [
                [[self.X_TURN, self.Y_START, self.Z_TURN],[self.X_TURN, self.Y_START, self.Z_TURN],[self.X_START, self.Y_TURN, self.Z_TURN],[self.X_START, self.Y_TURN,self.Z_TURN]],
                [[self.X_TURN, self.Y_START, self.Z_PUSH],[self.X_TURN, self.Y_START, self.Z_PUSH],[self.X_START, self.Y_TURN, self.Z_TURN],[self.X_START, self.Y_TURN,self.Z_TURN]],
                [[self.X_TURN, self.Y_START, self.Z_TURN],[self.X_TURN, self.Y_START, self.Z_TURN],[self.X_START, self.Y_TURN, self.Z_TURN],[self.X_START, self.Y_TURN,self.Z_TURN]],
                [[self.X_TURN, self.Y_START, self.Z_PUSH],[self.X_TURN, self.Y_START, self.Z_PUSH],[self.X_START, self.Y_TURN, self.Z_TURN],[self.X_START, self.Y_TURN,self.Z_TURN]],
                [[self.X_TURN, self.Y_START, self.Z_TURN],[self.X_TURN, self.Y_START, self.Z_TURN],[self.X_START, self.Y_TURN, self.Z_TURN],[self.X_START, self.Y_TURN,self.Z_TURN]],
                [[self.X_TURN, self.Y_START, self.Z_PUSH],[self.X_TURN, self.Y_START, self.Z_PUSH],[self.X_START, self.Y_TURN, self.Z_TURN],[self.X_START, self.Y_TURN,self.Z_TURN]],
                [[self.X_TURN, self.Y_START, self.Z_TURN],[self.X_TURN, self.Y_START, self.Z_TURN],[self.X_START, self.Y_TURN, self.Z_TURN],[self.X_START, self.Y_TURN,self.Z_TURN]],
            ]
            if self.stand_position == 0:
                _push_up.append([[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_TURN, self.Y_START,self.z_current],[self.X_TURN, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]])
            else:
                _push_up.append([[self.X_TURN, self.Y_START,self.z_current], [self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_TURN, self.Y_START, self.z_current]])
            return _push_up
        
        @property
        @check_stand
        @normal_action(0)
        def wave(self):
            return [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_TURN, self.Y_START,self.Z_UP],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_START, self.Y_WAVE,self.Z_WAVE],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_START, self.Y_WAVE,self.Z_UP],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_START, self.Y_WAVE,self.Z_WAVE],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_START, self.Y_WAVE,self.Z_UP],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_START, self.Y_WAVE,self.Z_WAVE],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_START, self.Y_WAVE,self.Z_UP],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_TURN, self.Y_START,self.Z_UP],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START,self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
            ]
        
        @property
        @check_stand
        @normal_action(1)
        def look_left(self):
            li = self.turn_angle_coord(self.angle)
            temp_x1 = li[0:2]
            temp_x1.append(self.z_current)
            temp_x2 = li[2:4]
            temp_x2.append(self.z_current)
            temp_x3 = li[4:6]
            temp_x3.append(self.z_current)
            return [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START,self.z_current],[self.X_TURN, self.Y_START, self.Z_UP],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [temp_x1, temp_x2,[self.X_TURN, self.Y_START, self.Z_UP],temp_x3]
            ]
            
        @property
        @check_stand
        @normal_action(1)
        def look_right(self):
            li = self.turn_angle_coord(self.angle)
            temp_x1 = li[0:2]
            temp_x1.append(self.z_current)
            temp_x2 = li[2:4]
            temp_x2.append(self.z_current)
            temp_x3 = li[4:6]
            temp_x3.append(self.z_current)
            return [
                [
                    [self.X_DEFAULT, self.Y_DEFAULT, self.z_current],
                    [self.X_TURN, self.Y_START,self.Z_UP],
                    [self.X_DEFAULT, self.Y_START, self.z_current],
                    [self.X_DEFAULT, self.Y_DEFAULT, self.z_current]
                ],
                [temp_x3, [self.X_TURN, self.Y_START, self.Z_UP], temp_x2, temp_x1]
            ]
        
        @property
        @check_stand
        @normal_action(1)
        def turn_left_angle(self):
            li = self.turn_angle_coord(self.angle)
            temp_x1 = li[0]
            temp_y1 = li[1]
            temp_x2 = li[2]
            temp_y2 = li[3]
            temp_x3 = li[4]
            temp_y3 = li[5]
            return [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START,self.z_current],[self.X_TURN, self.Y_START, self.Z_UP],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[temp_x1, temp_y1, self.z_current], [temp_x2, temp_y2, self.z_current],[self.X_TURN, self.Y_START, self.Z_UP],[temp_x3, temp_y3, self.z_current]],
                [[temp_x1, temp_y1, self.z_current], [temp_x2, temp_y2, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[temp_x3, temp_y3, self.z_current]],
                [[temp_x1, temp_y1, self.z_current], [temp_x2, temp_y2, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[temp_x3, temp_y3, self.Z_UP]],
                [[temp_x1, temp_y1, self.z_current], [temp_x2, temp_y2, self.z_current],[self.X_TURN, self.Y_DEFAULT, self.z_current],[self.X_TURN, self.Y_START, self.Z_UP]],
                [[temp_x1, temp_y1, self.z_current], [temp_x2, temp_y2, self.z_current],[self.X_TURN, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_START, self.z_current]]
            ]
            
        @property
        @check_stand
        @normal_action(1)
        def turn_right_angle(self):
            li = self.turn_angle_coord(self.angle)
            temp_x1 = li[0]
            temp_y1 = li[1]
            temp_x2 = li[2]
            temp_y2 = li[3]
            temp_x3 = li[4]
            temp_y3 = li[5]
            return [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_TURN, self.Y_START,self.Z_UP],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
                [[temp_x3,temp_y3, self.z_current], [self.X_TURN, self.Y_START, self.Z_UP], [temp_x2, temp_y2, self.z_current], [temp_x1, temp_y1, self.z_current]],
                [[temp_x3,temp_y3, self.z_current], [self.X_DEFAULT, self.Y_DEFAULT, self.z_current], [temp_x2, temp_y2, self.z_current], [temp_x1, temp_y1, self.z_current]],
                [[temp_x3,temp_y3, self.Z_UP], [self.X_DEFAULT, self.Y_DEFAULT, self.z_current], [temp_x2, temp_y2, self.z_current], [temp_x1, temp_y1, self.z_current]],
                [[self.X_TURN, self.Y_START, self.Z_UP], [self.X_DEFAULT, self.Y_DEFAULT, self.z_current], [temp_x2, temp_y2, self.z_current], [temp_x1, temp_y1, self.z_current]],
                [[self.X_DEFAULT, self.Y_START, self.z_current], [self.X_DEFAULT, self.Y_DEFAULT, self.z_current], [temp_x2, temp_y2, self.z_current], [temp_x1, temp_y1, self.z_current]],
            ]
            
        
        @property
        @check_stand
        @normal_action(0)
        def look_up(self):
            return [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.Z_DEFAULT],[self.X_DEFAULT, self.Y_START,self.Z_DEFAULT],[self.X_TURN, self.Y_START, self.Z_UP],[self.X_DEFAULT, self.Y_DEFAULT, self.Z_UP]],
            ]
            
        @property
        @check_stand
        @normal_action(0)
        def look_down(self):
            return [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.Z_UP],[self.X_TURN, self.Y_START,self.Z_UP],[self.X_DEFAULT, self.Y_START, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
            ]
        
        def rotate_body_absolute_x(self, degree_x):
            degree_x = degree_x * math.pi / 180
            dz = (self.LENGTH_SIDE / 2 + self.Y_DEFAULT) * math.sin(degree_x)
            dy = (self.LENGTH_SIDE / 2 + self.Y_DEFAULT) * (1 - math.cos(degree_x))
            return [[self.X_DEFAULT, self.Y_DEFAULT - dy, self.Z_DEFAULT + dz],[self.X_DEFAULT, self.Y_DEFAULT - dy, self.Z_DEFAULT - dz],[self.X_DEFAULT, self.Y_DEFAULT - dy, self.Z_DEFAULT - dz],[self.X_DEFAULT, self.Y_DEFAULT - dy, self.Z_DEFAULT + dz]]
        
        
        def rotate_body_absolute_y(self, degree_y):
            degree_y = degree_y * math.pi / 180
            dz = (self.LENGTH_SIDE / 2 + self.X_DEFAULT) * math.sin(degree_y)
            dx = (self.LENGTH_SIDE / 2 + self.X_DEFAULT) * (1 - math.cos(degree_y))
            # print("dz = %d"%dz)
            # print("dx = %d"%dx)
            return [[self.X_DEFAULT- dx, self.Y_DEFAULT, self.Z_DEFAULT + dz], [self.X_DEFAULT- dx, self.Y_DEFAULT, self.Z_DEFAULT + dz],[self.X_DEFAULT- dx, self.Y_DEFAULT, self.Z_DEFAULT - dz],[self.X_DEFAULT- dx, self.Y_DEFAULT, self.Z_DEFAULT - dz]]
        
        
        def  move_body_absolute(self, x, y, z):
            return [[self.X_DEFAULT - x,self.Y_DEFAULT - y,self.Z_TURN - z],[self.X_DEFAULT + x,self.Y_DEFAULT - y,self.Z_TURN - z],[self.X_DEFAULT + x,self.Y_DEFAULT + y,self.Z_TURN - z],[self.X_DEFAULT - x,self.Y_DEFAULT + y,self.Z_TURN - z]]
        
        
        def to_rad(self, deg):
            return deg * math.pi / 180
        
        @property
        def dance(self):
            _dance = []
            if not self.is_sit():
                _dance += self.sit
            _dance += [
                [[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current],[self.X_DEFAULT, self.Y_DEFAULT, self.z_current]],
            ]
            for i in range(0, 360, 5):
                _dance.append(self.move_body_absolute(40 * math.sin(self.to_rad(i)), 40 * math.cos(self.to_rad(i)), 0))
            for i in range(360, 0, -5):
                _dance.append(self.move_body_absolute(40 * math.sin(self.to_rad(i)), 40 * math.cos(self.to_rad(i)), 0))
            _dance.append(self.rotate_body_absolute_x(-20))
            _dance.append(self.rotate_body_absolute_x(20))
            _dance.append(self.move_body_absolute(0, 0, 0))
            _dance.append(self.rotate_body_absolute_y(-20))
            _dance.append(self.rotate_body_absolute_y(20))
            for j in range(0, 3):
                for i in range(0, 360, 3):
                    _dance.append(self.move_body_absolute(40 * math.sin(self.to_rad(i)), 40 * math.cos(self.to_rad(i)), (i / 360.0 + j) * 15))
            for j in range(3, 0, -1):
                for i in range(0, 360, 3):
                    _dance.append(self.move_body_absolute(40 * math.sin(self.to_rad(i)), 40 * math.cos(self.to_rad(i)), ((360 - i) / 360.0 + j - 1) * 15))
            _dance.append(self.move_body_absolute(0, 0, 0))
            return _dance



    def do_single_leg(self,leg,coodinate=[50,50,-33],speed=50):
        target_coord = self.current_step_all_leg_value()
        target_coord[leg] = coodinate
        self.do_step(target_coord,speed)
 

    def current_step_leg_value(self,leg):
        return list.copy(self.current_coord[leg])
        
    def current_step_all_leg_value(self):
        return list.copy(self.current_coord)

    def mix_step(self,basic_step,leg,coodinate=[50,50,-33]):
        # Pay attention to adding list(), otherwise the address pointer is returned
        new_step = list(basic_step)
        new_step[leg] = coodinate
        return list(new_step)

  
    
   
    
   