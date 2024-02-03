import dataclasses
from typing import Union


@dataclasses.dataclass(frozen=True)
class Problem(object):
    robot: Union[str, int]
    dynamic_object: Union[str, int]
    static_object: Union[str, int]
    init_insertable: tuple
    init_graspable: tuple
    init_placeable: tuple
    goal_inserted: tuple
    goal_holding: tuple
    goal_on: tuple
    cost: int
    body_names: dict

    def __repr__(self):
        return repr(self.__dict__)