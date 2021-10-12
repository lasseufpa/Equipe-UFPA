import rospy

from mrs_msgs.msg import PositionCommand, ReferenceStamped
from mrs_msgs.srv import ReferenceStampedSrv
from nav_msgs.msg import Odometry

from collector import Collector

class ControllerUtils(Collector):
    def __init__(self, odom_topic = '/uav1/odometry/odom_main'):
        super(ControllerUtils, self).__init__()
        self.odom = rospy.Subscriber(odom_topic, Odometry, self.callback)

        rospy.sleep(1)

    def _process(self, data):
        pose = data.pose.pose.position
        return [pose.x, pose.y, pose.z]
    
    def arrived(self, ref, eps=0.1):
        pos = self.data
        
        for i in range(len(pos)):
            if abs(pos[i] - ref[i]) > eps:
                return False

        return True
    
    # def split_trajectory(self, Pu, Pv, is_abs=False, split_factor=3.0):
    #     # Relative vector
    #     D = Pv
    #     if is_abs:
    #         for i in range(2):
    #             D[i] = Pv[i] - Pu[i]

    #     # Scaling
    #     for i in range(2):
    #         D[i] /= split_factor
        
    #     trajectory = [D for i in range(int(split_factor))]
    #     return trajectory


class Controller(Collector):
    def __init__(self,
                pose_topic = '/uav1/control_manager/position_cmd',
                control_srv = '/uav1/control_manager/reference'):
        super(Controller, self).__init__()
        self.tracker = rospy.Subscriber(pose_topic, PositionCommand, self.callback)
        self.control = rospy.ServiceProxy(control_srv, ReferenceStampedSrv, persistent=True)
        self.ref = ReferenceStamped()

        self.utils_arrived = ControllerUtils()

        rospy.sleep(1)

    def _update_reference(self):
        pose = self.data
        self.ref.reference.position = pose.position
        self.ref.reference.heading = pose.heading

    def _call(self):
        res = self.control(self.ref.header, self.ref.reference)
        rospy.sleep(1)

    # def _split_reference_pos(self, x, y, z, is_abs, kwargs):
    #     split_factor = 1.0
    #     if 'split_factor' in kwargs:
    #         split_factor = kwargs['split_factor']
    #     else: 
    #         split_factor = 3.0
        
    #     displacements = self.utils_arrived.split_trajectory(self.get_ref(), [x, y], split_factor=split_factor)

    #     if not is_abs:
    #         self.ref.reference.position.z += z
    #     else:
    #         self.ref.reference.position.z = z

    #     for delta in displacements:
    #         self.ref.reference.position.x += delta[0]
    #         self.ref.reference.position.y += delta[1]

    #         self._call()
    #         self.wait_to_arrive()

    def _fix_kwargs_pos(self, is_abs, kwargs):
        arr = ['x', 'y', 'z']
        ans = [0 for i in range(3)]
        ref = self.get_ref()
        for i in range(3):
            if arr[i] not in kwargs:
                ans[i] = ref[i]
            else:
                ans[i] = kwargs[arr[i]]                
        return ans
    
    def wait_to_arrive(self, eps=0.1):
        ref = [ 
            self.ref.reference.position.x,
            self.ref.reference.position.y,
            self.ref.reference.position.z
        ]
        while not self.utils_arrived.arrived(ref, eps=eps):
            rospy.sleep(0.1)

        rospy.logdebug('ARRIVED')
        rospy.sleep(3)

    def change_reference_pos(self, is_abs=False, **kwargs):
        rospy.logdebug('chaging reference position')
        self._update_reference()

        x, y, z = self._fix_kwargs_pos(is_abs=is_abs, kwargs=kwargs)

        if not is_abs:
            self.ref.reference.position.x += x
            self.ref.reference.position.y += y
            self.ref.reference.position.z += z
        else:
            self.ref.reference.position.x = x
            self.ref.reference.position.y = y
            self.ref.reference.position.z = z

        self._call()

        if 'arrive' in kwargs and kwargs['arrive']:
            self.wait_to_arrive()

    def change_reference_heading(self, h):
        rospy.logdebug('chaging reference heading')

        self.ref.reference.heading = h
        self._call()

    # CURRENT
    def get_ref(self):
        self._update_reference()
        return [
            self.ref.reference.position.x,
            self.ref.reference.position.y,
            self.ref.reference.position.z
        ]
    
    def center_at_base(self, centralize_srv, descend_factor=-0.01):
        rospy.loginfo('Starting centralizing')
        while True:
            data = centralize_srv()
            offset = data.offset
            any_base = data.any_base

            rospy.loginfo('Centralize? offset: %s; base: %s', str(offset), str(any_base))

            if any_base == False or (offset[0] == -10 and offset[1] == -10):
                return False
            
            if offset[0] == 0 and offset[1] == 0:
                break

            self.change_reference_pos(is_abs=False, x=offset[0], y=offset[1], z=descend_factor)
            
            rospy.sleep(3.5)

        return True
