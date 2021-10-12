#!/usr/bin/env python

import rospy
import argparse

from phase0_control.follow_trace import FollowTrace 

def get_args():
    parser = argparse.ArgumentParser(
        description='Follow bases using a trace file'
    )

    parser.add_argument('--trace',
                        help='path to trace file (.json)',
                        type=str,
                        default='../data/base_positions.json')
    
    parser.add_argument('--abs',
                        help='use abs position controls',
                        action='store_true')

    return parser.parse_args()

def main():
    args = get_args()

    rospy.init_node('follow_trace')

    control = FollowTrace(trace_file=args.trace)

    rospy.sleep(2)

    control.follow(is_abs=args.abs, eps=0.05)

if __name__ == '__main__':
    main()
