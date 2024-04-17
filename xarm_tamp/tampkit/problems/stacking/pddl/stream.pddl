(define (stream stacking-tamp)
  (:stream sample-grasp
    ; object ?o must be graspable and output ?g mush satisfy grasp condition

    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )

  (:stream sample-place
    ; object ?o mush be placeable on the surface ?r 
    ;   and output mush be stable on the surface ?r at pose ?p

    :inputs (?o1 ?o2)
    :domain (Placeable ?o1 ?o2)
    :outputs (?p)
    :certified (and (Pose ?o1 ?p) (RegionPose ?o2 ?p))
  )

  (:stream sample-insert
    ; sample insert pose
    ; object ?o mush be placeable on the surface ?r and output mush be stable on the surface ?r at pose ?p
    :inputs (?b1 ?b2)
    :domain (Insertable ?b1 ?b2)
    :outputs (?p)
    :certified (and (Pose ?b1 ?p) (HolePose ?b2 ?p))
  )

  (:stream sample-insert
    ; object ?o mush be placeable on the surface ?r 
    ;   and output mush be stable on the surface ?r at pose ?p

    :inputs (?o1 ?o2)
    :domain (Insertable ?o1 ?o2)
    :outputs (?p)
    :certified (and (Pose ?o1 ?p) (HolePose ?o2 ?p))
  )

  (:stream plan-motion
    :inputs (?a ?q1 ?q2)
    :domain (and (Arm ?a) (Conf ?q1) (Conf ?q2))
    :outputs (?t)
    :certified (and (Traj ?t) (Motion ?a ?q1 ?t ?q2))
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
    :certified (Supported ?o1 ?p1)
  )

  (:stream test-inserted
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (Assembled ?o1 ?p1)
  )

  (:function (MoveCost ?t)
    (and (Traj ?t))
  )
)