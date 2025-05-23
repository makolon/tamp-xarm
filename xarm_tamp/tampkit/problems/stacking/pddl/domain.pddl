(define (domain stacking-tamp)
  (:requirements :strips :equality)
  (:predicates
    ; Static predicates
    (Arm ?a)
    (Object ?o)
    (Region ?r)
    (Pose ?o ?p)
    (Grasp ?o ?g)
    (Traj ?t)
    (Conf ?q)
    (Kin ?o ?p ?g ?q ?t)
    (FreeMotion ?q1 ?t ?q2)
    (HoldingMotion ?q1 ?t ?q2 ?o ?g)
    (Graspable ?o)
    (Placeable ?o ?r)
    (CFreePosePose ?o ?p ?o2 ?p2)
    (CFreeApproachPose ?o ?p ?g ?o2 ?p2)
    (CFreeTrajPose ?t ?o2 ?p2)

    ; Fluent predicates
    (AtPose ?o ?p)
    (AtGrasp ?o ?g)
    (AtConf ?q)
    (CanMove)
    (HandEmpty)
    (Supported ?o ?p ?r)

    ; Derived predicates
    (On ?o ?r)
    (Holding ?o)
    (UnsafePose ?o ?p)
    (UnsafeApproach ?o ?p ?g)
    (UnsafeTraj ?t)
  )

  (:action move_free
    :parameters (?q1 ?q2 ?t)
    :precondition (and (CanMove)
                       (HandEmpty)
                       (AtConf ?q1)
                       (FreeMotion ?q1 ?t ?q2))
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (CanMove)))
  )

  (:action move_holding
    :parameters (?q1 ?q2 ?o ?g ?t)
    :precondition (and (CanMove)
                       (AtConf ?q1)
                       (AtGrasp ?o ?g)
                       (HoldingMotion ?q1 ?t ?q2 ?o ?g))
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (CanMove)))
  )

  (:action pick
    :parameters (?o ?p ?g ?q ?t)
    :precondition (and (HandEmpty)
                       (Kin ?o ?p ?g ?q ?t)
                       (AtPose ?o ?p)
                       (AtConf ?q)
                       (not (UnsafeApproach ?o ?p ?g))
                       (not (UnsafeTraj ?t)))
    :effect (and (CanMove)
                 (AtGrasp ?o ?g)
                 (not (AtPose ?o ?p))
                 (not (HandEmpty)))
  )

  (:action place
    :parameters (?o ?p ?g ?q ?t)
    :precondition (and (Kin ?o ?p ?g ?q ?t)
                       (AtGrasp ?o ?g)
                       (AtConf ?q)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafeApproach ?o ?p ?g))
                       (not (UnsafeTraj ?t)))
    :effect (and (CanMove)
                 (HandEmpty)
                 (AtPose ?o ?p)
                 (not (AtGrasp ?o ?g)))
  )

  (:derived (On ?o ?r)
    (exists (?p) (and (Supported ?o ?p ?r)
                      (AtPose ?o ?p)))
  )

  (:derived (Holding ?o)
    (exists (?g) (and (Grasp ?o ?g)
                      (AtGrasp ?o ?g)))
  )

  (:derived (UnsafePose ?o ?p)
    (exists (?o2 ?p2) (and (Pose ?o ?p)
                           (Pose ?o2 ?p2)
                           (AtPose ?o2 ?p2)
                           (not (= ?o ?o2))
                           (not (CFreePosePose ?o ?p ?o2 ?p2))))
  )

  (:derived (UnsafeApproach ?o ?p ?g)
    (exists (?o2 ?p2) (and (Grasp ?o ?g)
                           (Pose ?o ?p)
                           (Pose ?o2 ?p2)
                           (AtPose ?o2 ?p2)
                           (not (= ?o ?o2))
                           (not (CFreeApproachPose ?o ?p ?g ?o2 ?p2))))
  )

  (:derived (UnsafeTraj ?t)
    (exists (?o2 ?p2) (and (Traj ?t)
                           (Pose ?o2 ?p2)
                           (AtPose ?o2 ?p2)
                           (not (CFreeTrajPose ?t ?o2 ?p2))))
  )
)