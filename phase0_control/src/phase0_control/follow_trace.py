import rospy
import json

from controller import Controller

class FollowTrace(Controller):
    def __init__(self, trace_file):
        super(FollowTrace, self).__init__()
        self.trace = self._read_trace_file(trace_file)

    def _read_trace_file(self, trace_file):
        with open(trace_file, 'r') as f:
            return json.load(f)

    def follow(self, is_abs=False, eps=0.1):
        rospy.loginfo('Starting following trace')

        for vector in self.trace:
            self.change_reference_pos(is_abs=is_abs, x=vector[0], y=vector[1], z=vector[2])
            
            if is_abs:
                rospy.logdebug('Moving to position (%f, %f, %f)', vector[0], vector[1], vector[2])    
            else:
                rospy.logdebug('Moving to relative position (%f, %f, %f)', vector[0], vector[1], vector[2])    
                
            while not self.utils_arrived.arrived(self.get_ref(), eps=eps):
                rospy.sleep(0.2)
            
def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Control the drone with a trace file'
    )

    parser.add_argument('--trace',
                        help='path to trace file (.json)',
                        type=str,
                        default='./data/test_trace.json')
    
    return parser.parse_args()

def main():
    rospy.init_node('follow_trace')

    args = get_args()

    path = FollowTrace(trace_file=args.trace)
    rospy.sleep(3)
    
    path.follow()

if __name__ == '__main__':
    main()
