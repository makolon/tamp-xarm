(define (problem fetch-problem)
  (:domain fetch-tamp)
  
  (:objects
    arm1
    obj1
    region1
    region2
    pose1
    pose2
    grasp1
    traj1
    conf1
    conf2
  )

  (:init
    ; Static facts
    (Arm arm1)
    (Object obj1)
    (Region region1)
    (Region region2)
    (Pose obj1 pose1)
    (Pose obj1 pose2)
    (Grasp obj1 grasp1)
    (Traj traj1)
    (Conf conf1)
    (Conf conf2)
    (Kin arm1 obj1 pose1 grasp1 conf1 traj1)
    (Kin arm1 obj1 pose2 grasp1 conf2 traj1)
    (Motion arm1 conf1 traj1 conf2)
    (Graspable obj1)
    (Placeable obj1 region2)

    ; Initial fluent facts
    (AtPose obj1 pose1)
    (AtConf arm1 conf1)
    (HandEmpty arm1)

    ; Collision-free facts
    (CFreeTrajPose traj1 obj1 pose1)
    (CFreeTrajPose traj1 obj1 pose2)
    (CFreePosePose obj1 pose1 obj1 pose2)

    ; Cost functions
    (= (MoveCost traj1) 5)
    (= (PickCost) 10)
    (= (PlaceCost) 15)
    (= (total-cost) 0)
  )

  (:goal
    (and
      (AtPose obj1 pose2)
      (HandEmpty arm1)
    )
  )

  (:metric minimize (total-cost))
)