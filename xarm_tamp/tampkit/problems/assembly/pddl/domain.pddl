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
    (Controllable ?a)
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
    (CanMove)
    (HandEmpty ?a)
    (Placed ?o)
    (Inserted ?o)

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
    ;  - ?t  : trajectory from q1 to q2
    ; Precond:
    ;  - trajectory ?t must be motion from q1 to q2
    ;  - arm ?a must be initial configuration ?q1
    ;  - arm must be able to move
    ;  - trajectory ?t must be safe
    ; Effect:
    ;  - arm ?a configuration at the end must be ?q2
    ;  - arm ?a configuration at the end must not be ?q1
    ;  - configuration ?q1 is not ?q2

    :parameters (?q1 ?q2 ?t)
    :precondition (and (Motion ?a ?q1 ?t ?q2)
                       (AtConf ?a ?q1)
                       (CanMove)
                       (not (UnSafeTraj ?t)))
    :effect (and (AtConf ?a ?q2)
                 (not (AtConf ?a ?q1))
                 (not (= ?q1 ?q2))
                 (increase (total-cost) (MoveCost ?t)))
  )

  (:action pick
    ; Args:
    ;  - ?a : arm
    ;  - ?o : object
    ;  - ?p : object pose
    ;  - ?g : grasp pose
    ;  - ?q : arm configuration
    ;  - ?t : pick trajectory

    :parameters (?a ?o ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t)           ; kinematic constraints for pick pose ?p
                       (HandEmpty ?a)                    ; hand-empty constraints
                       (Graspable ?o)                    ; block ?o is graspable
                       (AtConf ?a ?q)                    ; base position is at conf ?q
                       (AtPose ?o ?p)                    ; block ?o pose is at pose ?p
                       (not (CanMove))                   ; cannot move
                       (not (UnsafeApproach ?o ?p ?g))   ; pose ?p with grasp ?g is not unsafe
                       (not (UnsafeTraj ?t)))            ; trajectory ?t is not unsafe
    :effect (and (CanMove)                               ; can move
                 (AtGrasp ?a ?o ?g)                      ; block ?o is at grasp ?g
                 (not (HandEmpty ?a))                    ; hand is not empty
                 (increase (total-cost) (PickCost)))     ; increase total-cost
  )

  (:action place
    ; Args:
    ;  - ?a  : arm
    ;  - ?o1 : object 1
    ;  - ?o2 ; object 2
    ;  - ?p  : object pose
    ;  - ?g  : grasp pose
    ;  - ?q  : arm configuration
    ;  - ?t  : pick trajectory

    :parameters (?a ?o1 ?o2 ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o1 ?p ?g ?q ?t)          ; kinematic constraints for place pose
                       (AtConf ?a ?q)                    ; base position is at conf ?q
                       (AtGrasp ?a ?o1 ?g)               ; at-grasp constraints
                       (Placeable ?o1 ?o2)               ; block b1 can place on block b2
                       (RegionPose ?o2 ?p)               ; region ?o2 is region-pose ?p
                       (not (CanMove))                   ; cannot move
                       (not (Placed ?o1))                ; block b1 is not placed yet
                       (not (AtPose ?o1 ?p))             ; block ?o1 is not at place pose ?p
                       (not (UnsafePose ?o1 ?p))         ; pose ?p is not unsafe
                       (not (UnsafeApproach ?o1 ?p ?g))  ; pose ?p with grasp ?g is not unsafe
                       (not (UnsafeTraj ?t)))            ; trajectory ?t is not unsafe
    :effect (and (AtPose ?o1 ?p)                         ; lock ?o1 is at place-pose ?p
                 (Placed ?o1)                            ; block ?o1 is placed
                 (increase (total-cost) (PlaceCost)))    ; increase total-cost
  )

  (:action insert
    ; Args:
    ;  - ?a  : arm
    ;  - ?o1 : object 1
    ;  - ?o2 ; object 2
    ;  - ?p  : object pose
    ;  - ?g  : grasp pose
    ;  - ?q  : arm configuration
    ;  - ?t  : pick trajectory

    :parameters (?a ?o1 ?o2 ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o1 ?p ?g ?q ?t)          ; kinematic constraints for insert pose
                       (AtConf ?a ?q)                    ; base position is at conf ?q1
                       (AtGrasp ?a ?o1 ?g)               ; at-grasp constraints
                       (Placed ?o1)                      ; block ?o1 is placed already
                       (Inserted ?o2)                    ; hole ?o2 is inserted already
                       (Insertable ?o1 ?o2)              ; block ?o1 can insert into hole ?o2
                       (AtPose ?o1 ?p1)                  ; block ?o1 is at-pose place pose ?p1
                       (HolePose ?o2 ?p2)                ; hole ?o2 is hole-pose ?p2
                       (not (CanMove))                   ; cannot move
                       (not (AtPose ?o1 ?p2))            ; block ?o1 is not at insert-pose ?p2
                       (not (Inserted ?o1))              ; block ?o1 is not inserted yet
                       (not (UnsafePose ?o1 ?p2))        ; pose ?p2 is not unsafe
                       (not (UnsafeApproach ?o1 ?p2 ?g)) ; pose ?p2 with grasp ?g is not unsafe
                       (not (UnsafeTraj ?t)))            ; trajectory ?t is not unsafe
    :effect (and (CanMove)                               ; can move
                 (Inserted ?o1)                          ; block ?o1 is inserted
                 (AtPose ?o1 ?p2)                        ; block ?o1 is at insert-pose ?p2
                 (HandEmpty ?a)                          ; hand is empty
                 (increase (total-cost) (InsertCost)))   ; increase total-cost
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

  (:derived (UnsafeATraj ?t)
    (exists (?o2 ?p2) (and (ATraj ?t)
                           (Pose ?o2 ?p2)
                           (not (CFreeTrajPose ?t ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )
  (:derived (UnsafeBTraj ?t)
    (or (exists (?o2 ?p2) (and (TrajPoseCollision ?t ?o2 ?p2)
                               (AtPose ?o2 ?p2)))
        (exists (?a ?q) (and (TrajArmCollision ?t ?a ?q)
                             (AtAConf ?a ?q)))
        (exists (?a ?o ?g) (and (TrajGraspCollision ?t ?a ?o ?g)
                                (AtGrasp ?a ?o ?g)))
    )
  )
)