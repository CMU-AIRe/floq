from agents.fql import FQLAgent
from agents.floq import FloQAgent
from agents.floq_distributional import FloQDistributionalAgent
from agents.floq_staleness import FloQStalenessAgent
from agents.fql_target_noise import FQLTargetNoiseAgent
from agents.floq_target_noise import FloQTargetNoiseAgent
from agents.fql_plasticity import FQLPlasticityAgent
from agents.floq_plasticity import FloQPlasticityAgent
from agents.fql_resnet_plasticity import FQLPlasticityResNetAgent
from agents.fql_transformer_plasticity import FQLPlasticityTransformerAgent
from agents.floq_predict_final import FloQPredictFinalAgent
from agents.fql_feature_norm import FQLFeatureNormAgent
from agents.floq_feature_norm import FloQFeatureNormAgent

agents = dict(
    fql=FQLAgent,
    floq=FloQAgent,
    floqdistributional=FloQDistributionalAgent,
    floqstaleness=FloQStalenessAgent,
    floqtargetnoise=FloQTargetNoiseAgent,
    fqltargetnoise=FQLTargetNoiseAgent,
    fqlplasticity=FQLPlasticityAgent,
    floqplasticity=FloQPlasticityAgent,
    fqlplasticityresnet=FQLPlasticityResNetAgent,
    fqlplasticitytransformer=FQLPlasticityTransformerAgent,
    floqpredictfinal=FloQPredictFinalAgent,
    fqlfeaturenorm=FQLFeatureNormAgent,
    floqfeaturenorm=FloQFeatureNormAgent,
)