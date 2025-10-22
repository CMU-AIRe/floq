from agents.fql import FQLAgent
from agents.fql_iqn import FQLIQNAgent
from agents.floq import FloQAgent


agents = dict(
    fql=FQLAgent,
    fqliqn = FQLIQNAgent,
    floq = FloQAgent,
)