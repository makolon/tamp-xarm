(define (stream xarm-assembly)
  (:stream sample-grasp
    :inputs (?r ?o)
    :domain (and (Robot ?r) (Graspable ?o))
    :outputs (?g)
    :certified (Grasp ?r ?o ?g))

  (:stream s-region
    :inputs (?b ?r)
    :domain (and (Placeable ?b ?r) (Block ?b) (Region ?r))
    :outputs (?p)
    :certified (and (Pose ?b ?p) (Contain ?b ?p ?r)))

  (:stream s-ik
    :inputs (?r ?b ?p ?g);(?b ?p ?g)
    :domain (and (Pose ?b ?p) (Grasp ?r ?b ?g) (Robot ?r))
    :outputs (?q ?t)
    :certified (and (Conf ?q) (Traj ?t) (Kin ?r ?b ?p ?g ?q ?t))) ;(Kin ?r ?b ?q ?p ?g)))

  (:stream s-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :outputs (?t)
    :certified (and (Traj ?t) (Motion ?q1 ?q2 ?t)))

  (:stream t-cfree
    :inputs (?b1 ?p1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Pose ?b2 ?p2))
    :certified (CFree ?b1 ?p1 ?b2 ?p2))
  
  (:stream test-cfree-traj-pose
    :inputs (?t ?o2 ?p2)
    :domain (and (Traj ?t) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?t ?o2 ?p2))

  (:stream s-blockregion
    :inputs (?bu ?bl ?pl)
    :domain (and (Pose ?bl ?pl) (Placeable ?bu ?bl))
    :outputs (?pu)
    :certified (and (Pose ?bu ?pu) (BlockContain ?bu ?pu ?bl ?pl)))

  (:stream t-cstack
    :inputs (?b1 ?p1 ?b2 ?p2)
    :domain (and (Pose ?b1 ?p1) (Pose ?b2 ?p2))
    :certified (CStack ?b1 ?p1 ?b2 ?p2))

  (:function (Dist ?q1 ?q2)
    (and (Conf ?q1) (Conf ?q2))) ; TODO: augment this with the keyword domain)

  (:function (Duration ?t)
             (Traj ?t))
)