(define (stream fetch-tamp)
  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  (:stream sample-place
    :inputs (?o1 ?o2)
    :domain (Placeable ?o1 ?o2)
    :outputs (?p)
    :certified (and (Pose ?o1 ?p) (Supported ?o1 ?p ?o2))
  )

  (:stream inverse-kinematics
    :inputs (?o ?p ?g)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q ?t)
    :certified (and (Conf ?q) (Traj ?t) (Kin ?o ?p ?g ?q ?t))
  )

  (:stream plan-free-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :outputs (?t)
    :certified (FreeMotion ?q1 ?t ?q2)
  )

  (:stream plan-holding-motion
    :inputs (?q1 ?q2 ?o ?g)
    :domain (and (Conf ?q1) (Conf ?q2) (Grasp ?o ?g))
    :outputs (?t)
    :certified (HoldingMotion ?q1 ?t ?q2 ?o ?g)
  )

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  )

  (:stream test-cfree-approach-pose
    :inputs (?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?o1 ?g1) (Pose ?o2 ?p2))
    :certified (CFreeApproachPose ?o1 ?p1 ?g1 ?o2 ?p2)
  )

  (:stream test-cfree-traj-pose
    :inputs (?t ?o ?p)
    :domain (and (Traj ?t) (Pose ?o ?p))
    :certified (CFreeTrajPose ?t ?o ?p)
  )

  (:stream test-supported
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (Supported ?o1 ?p1 ?o2)
  )

  (:stream test-inserted
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (Assembled ?o1 ?p1)
  )
)