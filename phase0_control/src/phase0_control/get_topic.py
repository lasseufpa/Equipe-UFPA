import rospy

from collector import Collector

class GetTopic(Collector):
    def __init__(self, topic, msg):
        super(GetTopic, self).__init__()
        self.topic = rospy.Subscriber(topic, msg, self.callback)

        rospy.sleep(1)

    def get(self):
        return self.data

if __name__ == '__main__':
    rospy.init_node('get_topic')
    """
        Given a topic like /uav1/odometry/odom_main
        To get what type of msgs it is begin published use rostopic info /uav1/odometry/odom_main
        The output will something like 

        Type: nav_msgs/Odometry

        Publishers: 
        * /uav1/uav1_nodelet_manager (http://ubuntu1804:42501/)

        Subscribers: 
        * /uav1/uav1_nodelet_manager (http://ubuntu1804:42501/)

        The message type is nav_msgs/Odometry, import it to the program with

        from nav_msgs.msg import Odometry

        There you go. The procedure is similar to every topic.
    """

    from nav_msgs.msg import Odometry
    print(Odometry)
    odom = GetTopic(topic='/uav1/odometry/odom_main', msg=Odometry)
    rospy.sleep(1)
    print(odom.get())