(define (problem fetch-problem)
  (:domain fetch-tamp)
  (:objects
    robot_arm - arm
    block1 block2 - object
    table1 table2 - region
    pose1 pose2 - pose
    grasp1 grasp2 - grasp
    traj1 traj2 - traj
    conf1 conf2 - conf
  )
  (:init
    (Arm robot_arm)
    (Object block1)
    (Object block2)
    (Region table1)
    (Region table2)
    (Pose block1 pose1)
    (Pose block2 pose2)
    (Pose block1 pose1)
    (Pose block2 pose2)
    (Grasp block1 grasp1)
    (Grasp block2 grasp2)
    (Traj traj1)
    (Traj traj2)
    (Conf conf1)
    (Conf conf2)

    ; define initial state
    (CanMove)
    (HandEmpty)
    (AtConf conf1)
    (Graspable block1)
    (Graspable block2)
    (Placeable block1 table1)
    (Placeable block2 table1)
    (Placeable block1 table2)
    (Placeable block2 table2)
    (AtPose block1 pose1)
    (AtPose block2 pose2)
    (AtPose block1 pose1)
    (AtPose block2 pose2)
    (Supported block1 pose1 table1)
    (Supported block2 pose2 table1)
    (Supported block1 pose1 table2)
    (Supported block2 pose2 table2)
    (Kin block1 pose1 grasp1 conf1 traj1)
    (Kin block2 pose2 grasp2 conf1 traj2)
    (FreeMotion conf1 traj1 conf2)
    (FreeMotion conf1 traj2 conf2)
    (HoldingMotion conf1 traj1 conf2 block1 grasp1)
    (HoldingMotion conf1 traj2 conf2 block2 grasp2)
  )
  (:goal
    (and
      (On block1 table2)
      (On block2 table2)
    )
  )
)