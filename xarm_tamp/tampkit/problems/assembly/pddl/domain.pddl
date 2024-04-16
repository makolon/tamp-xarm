(define (domain assembly-tamp)
  (:requirements :strips :equality)
  (:predicates
    ; Static predicates
    (Arm ?a)
    (Object ?o)
    (Region ?r)
    (Hole ?h)
    (Pose ?o ?p)
    (Grasp ?o ?g)
    (Traj ?t)
    (Conf ?q)
    (Kin ?a ?o ?p ?g ?q ?t)
    (Motion ?a ?q1 ?t ?q2)
    (Graspable ?o)
    (Placeable ?o ?r)
    (Insertable ?o ?h)
    (RegionPose ?r ?p)
    (HolePose ?h ?p)
    (TrajPoseCollision ?t ?o ?p)
    (TrajArmCollision ?t ?a ?q)
    (TrajGraspCollision ?t ?a ?o ?g)
    (CFreeTrajPose ?t ?o ?p)
    (CFreePosePose ?o1 ?p1 ?o2 ?p2)
    (CFreeApproachPose ?o1 ?p1 ?g ?o2 ?p2)

    ; Fluent predicates
    (AtPose ?o ?p)
    (AtGrasp ?a ?o ?g)
    (AtConf ?a ?q)
    (CanMove ?a)
    (HandEmpty ?a)
    (Placed ?o)
    (Inserted ?o)
    (Supported ?o ?p)
    (Assembled ?o ?p)

    ; Derived predicates
    (On ?o ?r)
    (InHole ?o ?h)
    (Holding ?a ?o)
    (UnsafePose ?o ?p)
    (UnsafeApproach ?o ?p ?g)
    (UnsafeTraj ?t)
  )

  (:functions
    (MoveCost ?t)
    (PickCost)
    (PlaceCost)
    (InsertCost)
  )

  (:action move
    ; Args:
    ;  - ?q1 : initial configuration
    ;  - ?q2 : final configuration
    ;  - ?t  : trajectory from ?q1 to ?q2
    ; Precond:
    ;  - The trajectory ?t must be a motion from ?q1 to ?q2
    ;  - The arm ?a must be in the initial configuration ?q1
    ;  - The trajectory ?t must be safe
    ; Effect:
    ;  - The configuration of arm ?a at the end must be ?q2
    ;  - The configuration of arm ?a at the end must not be ?q1
    ;  - The configuration ?q1 must not be the same as ?q2

    :parameters (?a ?q1 ?q2 ?t)
    :precondition (and (Motion ?a ?q1 ?t ?q2)
                       (AtConf ?a ?q1)
                       (not (UnSafeTraj ?t)))
    :effect (and (AtConf ?a ?q2)
                 (not (AtConf ?a ?q1))
                 (not (= ?q1 ?q2))
                 (increase (total-cost) (MoveCost ?t)))
  )

  (:action pick
    ; Args:
    ;  - ?a : the arm
    ;  - ?o : the object to be picked
    ;  - ?p : pose of the object ?o
    ;  - ?g : grasp type
    ;  - ?q : configuration of the arm
    ;  - ?t : trajectory
    ; Precond:
    ;  - Kinematic constraints ?a, ?o, ?p, ?g, ?q, ?t must be satisfied
    ;  - The hand of arm ?a must be empty
    ;  - Object ?o must be graspable
    ;  - Arm ?a must be at configuration ?q
    ;  - Object ?o must be at pose ?p
    ;  - The approach to object ?o at pose ?p must be safe
    ;  - The trajectory ?t must be safe
    ; Effect:
    ;  - The arm ?a is grasping object ?o with grasp ?g
    ;  - The hand of arm ?a is no longer empty
    ;  - The total cost is increased by the picking cost

    :parameters (?a ?o ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t)
                       (HandEmpty ?a)
                       (Graspable ?o)
                       (AtConf ?a ?q)
                       (AtPose ?o ?p)
                       (not (UnsafeApproach ?o ?p ?g))
                       (not (UnsafeTraj ?t)))
    :effect (and (AtGrasp ?a ?o ?g)
                 (not (HandEmpty ?a))
                 (increase (total-cost) (PickCost)))
  )

  (:action place
    ; Args:
    ;  - ?a : the arm
    ;  - ?o1 : the object to be placed
    ;  - ?o2 : the target object or area
    ;  - ?p : target pose
    ;  - ?g : grasp type used to hold ?o1
    ;  - ?q : configuration of the arm
    ;  - ?t : trajectory
    ; Precond:
    ;  - Kinematic constraints for ?a, ?o1, ?p, ?g, ?q, ?t must be satisfied
    ;  - Arm ?a must be at configuration ?q
    ;  - Arm ?a must be grasping object ?o1 with grasp ?g
    ;  - Object ?o1 must be placeable relative to ?o2
    ;  - Target region or pose for ?o2 must be ?p
    ;  - Object ?o1 must not already be placed
    ;  - Object ?o1 must not be at pose ?p
    ;  - The pose ?p for ?o1 must be safe
    ;  - The approach to pose ?p must be safe
    ;  - The trajectory ?t must be safe
    ; Effect:
    ;  - Object ?o1 is placed at pose ?p
    ;  - Object ?o1 is marked as placed
    ;  - The total cost is increased by the placement cost

    :parameters (?a ?o1 ?o2 ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o1 ?p ?g ?q ?t)
                       (AtConf ?a ?q)
                       (AtGrasp ?a ?o1 ?g)
                       (Placeable ?o1 ?o2)
                       (RegionPose ?o2 ?p)
                       (not (Placed ?o1))
                       (not (AtPose ?o1 ?p))
                       (not (UnsafePose ?o1 ?p))
                       (not (UnsafeApproach ?o1 ?p ?g))
                       (not (UnsafeTraj ?t)))
    :effect (and (AtPose ?o1 ?p)
                 (Placed ?o1)
                 (increase (total-cost) (PlaceCost)))
  )

  (:action insert
    ; Args:
    ;  - ?a : the arm
    ;  - ?o1 : the object to be inserted
    ;  - ?o2 : the target object or slot
    ;  - ?p : unused in preconditions and effects (presumed error)
    ;  - ?g : grasp type used to hold ?o1
    ;  - ?q : configuration of the arm
    ;  - ?t : trajectory
    ;  - ?p1 : current pose of ?o1 (assumed from AtPose ?o1 ?p1)
    ;  - ?p2 : target hole pose for ?o2
    ; Precond:
    ;  - Kinematic constraints for ?a, ?o1, ?p, ?g, ?q, ?t must be satisfied
    ;  - Arm ?a must be at configuration ?q
    ;  - Arm ?a must be grasping object ?o1 with grasp ?g
    ;  - Object ?o1 must be placed
    ;  - Object ?o2 must be inserted
    ;  - Object ?o1 must be insertable into ?o2
    ;  - Object ?o1 must be at pose ?p1
    ;  - Target hole for ?o2 must be at pose ?p2
    ;  - Object ?o1 must not already be at pose ?p2
    ;  - Object ?o1 must not already be inserted
    ;  - The pose ?p2 for ?o1 must be safe
    ;  - The approach to pose ?p2 must be safe
    ;  - The trajectory ?t must be safe
    ; Effect:
    ;  - Object ?o1 is marked as inserted
    ;  - Object ?o1 is at pose ?p2
    ;  - The hand of arm ?a is empty
    ;  - The total cost is increased by the insertion cost

    :parameters (?a ?o1 ?o2 ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o1 ?p ?g ?q ?t)
                       (AtConf ?a ?q)
                       (AtGrasp ?a ?o1 ?g)
                       (Placed ?o1)
                       (Inserted ?o2)
                       (Insertable ?o1 ?o2)
                       (AtPose ?o1 ?p1)
                       (HolePose ?o2 ?p2)
                       (not (AtPose ?o1 ?p2))
                       (not (Inserted ?o1))
                       (not (UnsafePose ?o1 ?p2))
                       (not (UnsafeApproach ?o1 ?p2 ?g))
                       (not (UnsafeTraj ?t)))
    :effect (and (Inserted ?o1)
                 (AtPose ?o1 ?p2)
                 (HandEmpty ?a)
                 (increase (total-cost) (InsertCost)))
  )

  (:derived (On ?o1 ?o2)
    (exists (?p) (and (Placed ?o1)
                      (AtPose ?o1 ?p)))
  )

  (:derived (InHole ?o1 ?o2)
    (exists (?p) (and (Inserted ?o1)
                      (AtPose ?o1 ?p)))
  )

  (:derived (Holding ?a ?o)
    (exists (?g) (and (Arm ?a) (Grasp ?o ?g)
                      (AtGrasp ?a ?o ?g)))
  )

  (:derived (UnsafePose ?o ?p)
    (exists (?o2 ?p2) (and (Pose ?o ?p) (Pose ?o2 ?p2)
                           (not (= ?o ?o2))
                           (not (CFreePosePose ?o ?p ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )

  (:derived (UnsafeApproach ?o ?p ?g)
    (exists (?o2 ?p2) (and (Grasp ?o ?g)
                           (Pose ?o ?p)
                           (Pose ?o2 ?p2)
                           (not (= ?o ?o2))
                           (not (CFreeApproachPose ?o ?p ?g ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )

  (:derived (UnsafeTraj ?t)
    (exists (?o2 ?p2) (and (Traj ?t)
                           (Pose ?o2 ?p2)
                           (not (CFreeTrajPose ?t ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )
  (:derived (UnsafeBTraj ?t)
    (or (exists (?o2 ?p2) (and (TrajPoseCollision ?t ?o2 ?p2)
                               (AtPose ?o2 ?p2)))
        (exists (?a ?q) (and (TrajArmCollision ?t ?a ?q)
                             (AtConf ?a ?q)))
        (exists (?a ?o ?g) (and (TrajGraspCollision ?t ?a ?o ?g)
                                (AtGrasp ?a ?o ?g)))
    )
  )
)